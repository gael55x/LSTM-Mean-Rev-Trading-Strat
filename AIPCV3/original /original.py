import os
import datetime
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI(
    title="Trading Strategy API",
    description="An API to run a mean reversion trading strategy with LSTM networks and test custom data.",
    version="1.0.0"
)

MODEL = None
SCALER = None
TRAINING_HISTORY = None  #  store training loss history
BACKTEST_RESULTS = None  # store backtest metrics and the equity curve
SEQ_LENGTH = 60

FEATURES = ['Close', 'pctB', 'RSI', 'MACD', 'Signal_Line', 'Momentum', 'ATR']

# ----------------------------
# Response models
# ----------------------------
class StrategyResponse(BaseModel):
    mae: float
    mse: float
    rmse: float
    num_buy_signals: int
    num_sell_signals: int
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    positions: List[dict]

class PredictRequest(BaseModel):
    # A sequence is expected to be a list of lists,
    # where each inner list represents one timestep with features in the order:
    sequence: List[List[float]]

class PredictResponse(BaseModel):
    predicted_close: float
    decision: str


def create_sequences(data_array, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data_array)):
        X.append(data_array[i - seq_length:i])
        y.append(data_array[i, 0])  # 0 corresponds to 'Close' after scaling
    return np.array(X), np.array(y)

def analyze_equity_curve(equity_curve: List[dict]) -> dict:
    """
    Analyze the equity curve to identify contiguous periods of good performance (positive pct change)
    and poor performance (negative pct change). Returns a dict with two lists.
    """
    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    segments = []
    current_segment = {'start': df.loc[0, 'date'], 'start_value': df.loc[0, 'portfolio_value']}
    current_sign = np.sign(df.loc[1, 'portfolio_value'] - df.loc[0, 'portfolio_value'])
    for i in range(1, len(df)):
        change = df.loc[i, 'portfolio_value'] - df.loc[i-1, 'portfolio_value']
        sign = np.sign(change)
        if sign == 0:  
            sign = current_sign
        if sign != current_sign:
            end_date = df.loc[i-1, 'date']
            end_value = df.loc[i-1, 'portfolio_value']
            pct_change = (end_value - current_segment['start_value']) / current_segment['start_value']
            segments.append({
                'start': current_segment['start'].strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'pct_change': pct_change
            })
            current_segment = {'start': df.loc[i, 'date'], 'start_value': df.loc[i, 'portfolio_value']}
            current_sign = sign
    end_date = df.loc[len(df)-1, 'date']
    end_value = df.loc[len(df)-1, 'portfolio_value']
    pct_change = (end_value - current_segment['start_value']) / current_segment['start_value']
    segments.append({
        'start': current_segment['start'].strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d'),
        'pct_change': pct_change
    })
    good_periods = [seg for seg in segments if seg['pct_change'] > 0]
    poor_periods = [seg for seg in segments if seg['pct_change'] < 0]
    return {'good_periods': good_periods, 'poor_periods': poor_periods}

def analyze_training_history(history: dict) -> dict:
    """
    Analyze the training history to identify the best and worst epochs based on validation loss.
    """
    val_loss = history.get('val_loss', [])
    if not val_loss:
        return {}
    best_epoch = int(np.argmin(val_loss) + 1)
    worst_epoch = int(np.argmax(val_loss) + 1)
    return {
        'best_epoch': best_epoch,
        'best_val_loss': val_loss[best_epoch - 1],
        'worst_epoch': worst_epoch,
        'worst_val_loss': val_loss[worst_epoch - 1]
    }

def run_trading_strategy():
    global MODEL, SCALER, TRAINING_HISTORY, BACKTEST_RESULTS, SEQ_LENGTH, FEATURES

    try:
        ticker = 'BTC-USD'
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=7 * 365) 
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        data = yf.download(
            tickers=ticker,
            start=start_date_str,
            end=end_date_str,
            interval='1d',
            progress=False
        )
        data.dropna(inplace=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            print("Flattened columns:", data.columns.tolist())

        if 'Close' not in data.columns and 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
            print("Using 'Adj Close' as 'Close'.")

        if data.empty or 'Close' not in data.columns:
            raise ValueError("No data downloaded or the 'Close' column is missing. Please check the ticker symbol and internet connection.")
        else:
            print("Data downloaded successfully.")


        if 'Close' in data.columns:
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()

            data['STD'] = data['Close'].rolling(window=20).std()
            data['Upper_Band'] = data['MA20'] + (data['STD'] * 2.5)
            data['Lower_Band'] = data['MA20'] - (data['STD'] * 2.5)

            close = data['Close'].squeeze()
            lower = data['Lower_Band'].squeeze()
            upper = data['Upper_Band'].squeeze()
            data['pctB'] = (close - lower) / (upper - lower)

            delta = data['Close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            roll_up = up.rolling(14).mean()
            roll_down = down.rolling(14).mean()
            RS = roll_up / roll_down
            data['RSI'] = 100.0 - (100.0 / (1.0 + RS))

            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

            data['Momentum'] = data['Close'] - data['Close'].shift(10)

            data['TR'] = data[['High', 'Close']].max(axis=1) - data[['Low', 'Close']].min(axis=1)
            data['ATR'] = data['TR'].rolling(window=14).mean()

            data.dropna(inplace=True)
            print("Data after preprocessing shape:", data.shape)
        else:
            raise ValueError("The 'Close' column is missing from the data.")

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[FEATURES])

        X, y = create_sequences(scaled_data, SEQ_LENGTH)
        if X.size == 0 or y.size == 0:
            raise ValueError("Insufficient data to create sequences. Try reducing the sequence length.")

        train_size = int(5 * 365)
        test_size = int(2 * 365)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:train_size + test_size]
        y_test = y[train_size:train_size + test_size]

        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True,
                       input_shape=(X_train.shape[1], X_train.shape[2]),
                       kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=64, return_sequences=False, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        print("Training complete!")

        TRAINING_HISTORY = history.history
        MODEL = model
        SCALER = scaler

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        zeros = np.zeros((predictions.shape[0], len(FEATURES) - 1))
        predictions_extended = np.hstack((predictions, zeros))
        actuals_extended = np.hstack((y_test.reshape(-1, 1), zeros))

        predicted_close = scaler.inverse_transform(predictions_extended)[:, 0]
        actual_close = scaler.inverse_transform(actuals_extended)[:, 0]

        test_data = data.iloc[train_size + SEQ_LENGTH: train_size + SEQ_LENGTH + test_size].copy()
        test_data['Predicted_Close'] = predicted_close
        test_data['Actual_Close'] = actual_close

        test_data['Predicted_Change'] = (test_data['Predicted_Close'] - test_data['Actual_Close']) / test_data['Actual_Close']
        test_data['Signal'] = 0

        rsi_buy_threshold = test_data['RSI'].quantile(0.4)
        rsi_sell_threshold = test_data['RSI'].quantile(0.6)
        predicted_change_buy_threshold = test_data['Predicted_Change'].quantile(0.6)
        predicted_change_sell_threshold = test_data['Predicted_Change'].quantile(0.4)

        test_data.loc[
            (test_data['Predicted_Change'] > predicted_change_buy_threshold) &
            (test_data['RSI'] < rsi_buy_threshold),
            'Signal'
        ] = 1

        test_data.loc[
            (test_data['Predicted_Change'] < predicted_change_sell_threshold) &
            (test_data['RSI'] > rsi_sell_threshold),
            'Signal'
        ] = -1

        num_buy_signals = int((test_data['Signal'] == 1).sum())
        num_sell_signals = int((test_data['Signal'] == -1).sum())

        ##############################
        # Simulate Trading with Risk Management
        ##############################
        initial_capital = 500.0
        positions = []
        cash = initial_capital
        holdings = 0
        portfolio_value = []
        transaction_cost = 0.0005
        stop_loss_percent = 0.1
        take_profit_percent = 0.2
        entry_price = None

        for index, row in test_data.iterrows():
            price = row['Actual_Close']
            signal = row['Signal']
            if signal == 1 and cash > 0:
                amount_to_buy = (cash * 0.5) * (1 - transaction_cost)
                holdings += amount_to_buy / price
                cash -= amount_to_buy
                entry_price = price
                positions.append({'Date': str(index), 'Position': 'Buy', 'Price': price})
            elif signal == -1 and holdings > 0:
                amount_to_sell = holdings * price * (1 - transaction_cost)
                cash += amount_to_sell
                holdings = 0
                entry_price = None
                positions.append({'Date': str(index), 'Position': 'Sell', 'Price': price})
            elif holdings > 0 and entry_price is not None:
                if price <= entry_price * (1 - stop_loss_percent):
                    amount_to_sell = holdings * price * (1 - transaction_cost)
                    cash += amount_to_sell
                    holdings = 0
                    positions.append({'Date': str(index), 'Position': 'Stop Loss Sell', 'Price': price})
                    entry_price = None
                elif price >= entry_price * (1 + take_profit_percent):
                    amount_to_sell = holdings * price * (1 - transaction_cost)
                    cash += amount_to_sell
                    holdings = 0
                    positions.append({'Date': str(index), 'Position': 'Take Profit Sell', 'Price': price})
                    entry_price = None

            total_value = cash + holdings * price
            portfolio_value.append(total_value)

        test_data['Portfolio_Value'] = portfolio_value[:len(test_data)]
        test_data['Daily_Return'] = test_data['Portfolio_Value'].pct_change()
        test_data['Cumulative_Return'] = (1 + test_data['Daily_Return']).cumprod()

        total_days = (test_data.index[-1] - test_data.index[0]).days
        total_days = total_days if total_days != 0 else 1

        annualized_return = (test_data['Cumulative_Return'].iloc[-1]) ** (365 / total_days) - 1
        returns = test_data['Daily_Return'].dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0
        rolling_max = test_data['Portfolio_Value'].cummax()
        drawdown = test_data['Portfolio_Value'] / rolling_max - 1
        max_drawdown = drawdown.min()
        total_return = ((test_data['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital) * 100

        result = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "num_buy_signals": num_buy_signals,
            "num_sell_signals": num_sell_signals,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "positions": positions
        }

        equity_curve = [{"date": str(date.date()), "portfolio_value": value}
                        for date, value in zip(test_data.index, test_data["Portfolio_Value"])]

        BACKTEST_RESULTS = {
            "metrics": result,
            "equity_curve": equity_curve
        }

        return result

    except Exception as e:
        print(f"Error occurred: {e}")
        raise e

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/run-strategy", response_model=StrategyResponse)
def run_strategy_endpoint():
    """
    Runs the trading strategy simulation (training and backtesting) and returns performance metrics.
    """
    try:
        result = run_trading_strategy()
        return result
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    """
    Predicts the next "Close" value based on a provided sequence of raw data and returns a trading decision.
    The decision is based on comparing the predicted next close with the last known close using a simple threshold.
    """
    global MODEL, SCALER, SEQ_LENGTH, FEATURES

    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=400, detail="Model is not trained yet. Please call the /run-strategy endpoint first.")

    sequence = payload.sequence
    if len(sequence) != SEQ_LENGTH:
        raise HTTPException(status_code=400, detail=f"Input sequence length must be {SEQ_LENGTH}.")

    for entry in sequence:
        if len(entry) != len(FEATURES):
            raise HTTPException(status_code=400, detail=f"Each entry must have {len(FEATURES)} features: {FEATURES}.")

    input_seq = np.array(sequence)
    scaled_seq = SCALER.transform(input_seq)
    scaled_seq = np.expand_dims(scaled_seq, axis=0)

    prediction = MODEL.predict(scaled_seq)
    zeros = np.zeros((prediction.shape[0], len(FEATURES) - 1))
    prediction_extended = np.hstack((prediction, zeros))
    predicted_close = SCALER.inverse_transform(prediction_extended)[:, 0]

    # Use the last known "Close" value from the input sequence (first feature)
    last_close = sequence[-1][0]
    threshold = 0.01
    if predicted_close[0] > last_close * (1 + threshold):
        decision = "Buy"
    elif predicted_close[0] < last_close * (1 - threshold):
        decision = "Sell"
    else:
        decision = "Hold"

    return PredictResponse(predicted_close=predicted_close[0], decision=decision)

@app.get("/backtest-details")
def backtest_details():
    """
    Returns detailed backtest results, including the equity curve.
    Run /run-strategy first to generate these details.
    """
    global BACKTEST_RESULTS
    if BACKTEST_RESULTS is None:
        raise HTTPException(status_code=400, detail="Backtest details not available. Please run /run-strategy first.")
    return BACKTEST_RESULTS

@app.get("/training-metrics")
def training_metrics():
    """
    Returns the training loss history (training and validation loss per epoch).
    Run /run-strategy first to generate these metrics.
    """
    global TRAINING_HISTORY
    if TRAINING_HISTORY is None:
        raise HTTPException(status_code=400, detail="Training metrics not available. Please run /run-strategy first.")
    return TRAINING_HISTORY

@app.get("/performance-periods")
def performance_periods():
    """
    Analyzes the backtest equity curve to identify periods of good and poor performance.
    Returns lists of good periods and poor periods with start date, end date, and percentage change.
    """
    global BACKTEST_RESULTS
    if BACKTEST_RESULTS is None or "equity_curve" not in BACKTEST_RESULTS:
        raise HTTPException(status_code=400, detail="Performance periods not available. Please run /run-strategy first.")
    periods = analyze_equity_curve(BACKTEST_RESULTS["equity_curve"])
    return periods

@app.get("/training-analysis")
def training_analysis():
    """
    Analyzes the training loss history and returns the epoch with the best (lowest) and worst (highest) validation loss.
    """
    global TRAINING_HISTORY
    if TRAINING_HISTORY is None:
        raise HTTPException(status_code=400, detail="Training analysis not available. Please run /run-strategy first.")
    analysis = analyze_training_history(TRAINING_HISTORY)
    return analysis

# ----------------------------
# Run the server (for local testing)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
