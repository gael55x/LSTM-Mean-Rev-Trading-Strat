import os
import warnings
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

from app.config import SEQ_LENGTH, FEATURES
from app.schema import StrategyResponse, PredictRequest, PredictResponse
from app.data_processing import download_data, preprocess_data, create_sequences, scale_data
from app.model_module import build_model, train_model, predict_value
from app.trading import simulate_trading
from app.analysis import analyze_equity_curve, analyze_training_history

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI(
    title="Trading Strategy API",
    description="An API to run a mean reversion trading strategy with LSTM networks and test custom data.",
    version="1.0.0"
)

# import the global state from config so that it can be updated
from app import config

def run_trading_strategy():
    try:
        # data download and preprocessing
        data = download_data()
        data = preprocess_data(data)
        print("data downloaded and preprocessed. shape:", data.shape)
        
        # we scale the data and form sequences
        scaled_data, scaler = scale_data(data)
        X, y = create_sequences(scaled_data, SEQ_LENGTH)
        # if sequences are empty, we raise an error
        if X.size == 0 or y.size == 0:
            raise ValueError("insufficient data to create sequences. try reducing the sequence length.")

        # define training and test split sizes (example: 5 years training, 2 years testing)
        train_size = int(5 * 365)
        test_size = int(2 * 365)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:train_size + test_size]
        y_test = y[train_size:train_size + test_size]

        # build and train the model
        model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        history = train_model(model, X_train, y_train, X_test, y_test)
        print("model training complete.")

        # store the training history and model/scaler in config for reuse
        config.TRAINING_HISTORY = history.history
        config.MODEL = model
        config.SCALER = scaler

        # predict on test data
        predictions = model.predict(X_test)
        # we calculate mean absolute error, mean squared error, and root mean squared error
        mae = float(np.mean(np.abs(y_test - predictions)))
        mse = float(np.mean((y_test - predictions) ** 2))
        rmse = float(np.sqrt(mse))

        # we rebuild arrays to inverse scale them properly
        zeros = np.zeros((predictions.shape[0], len(FEATURES) - 1))
        predictions_extended = np.hstack((predictions, zeros))
        actuals_extended = np.hstack((y_test.reshape(-1, 1), zeros))
        predicted_close = scaler.inverse_transform(predictions_extended)[:, 0]
        actual_close = scaler.inverse_transform(actuals_extended)[:, 0]

        # we slice the dataframe to match the test period after accounting for sequence length
        test_data = data.iloc[train_size + SEQ_LENGTH: train_size + SEQ_LENGTH + test_size].copy()
        
        # we run a backtest simulation on the predicted close prices vs the actual close
        backtest_results, _ = simulate_trading(test_data, predicted_close, actual_close)
        # we update the simulation metrics with our error measurements
        backtest_results["metrics"].update({
            "mae": mae,
            "mse": mse,
            "rmse": rmse
        })
        config.BACKTEST_RESULTS = backtest_results

        return backtest_results["metrics"]

    except Exception as e:
        print(f"error occurred: {e}")
        raise e

@app.get("/")
def health_check():
    # returns a simple status check
    return {"status": "OK"}

@app.get("/run-strategy", response_model=StrategyResponse)
def run_strategy_endpoint():
    # we run the strategy and return the results as a response model
    try:
        result = run_trading_strategy()
        return result
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    # we ensure the model is already trained
    if config.MODEL is None or config.SCALER is None:
        raise HTTPException(status_code=400, detail="model is not trained yet. please call the /run-strategy endpoint first.")

    sequence = payload.sequence
    # we check that the input sequence is the correct length
    if len(sequence) != SEQ_LENGTH:
        raise HTTPException(status_code=400, detail=f"input sequence length must be {SEQ_LENGTH}.")

    # we verify each entry in the sequence has the correct number of features
    for entry in sequence:
        if len(entry) != len(FEATURES):
            raise HTTPException(status_code=400, detail=f"each entry must have {len(FEATURES)} features: {FEATURES}.")

    # we get the predicted close using our trained model
    predicted_close = predict_value(config.MODEL, config.SCALER, sequence, FEATURES)
    # we compare it to the last known close in the sequence
    last_close = sequence[-1][0]
    threshold = 0.01

    # we create a decision rule based on a simple threshold
    if predicted_close > last_close * (1 + threshold):
        decision = "Buy"
    elif predicted_close < last_close * (1 - threshold):
        decision = "Sell"
    else:
        decision = "Hold"

    return PredictResponse(predicted_close=predicted_close, decision=decision)

@app.get("/backtest-details")
def backtest_details():
    # we return the stored backtest results or raise if none exist
    if config.BACKTEST_RESULTS is None:
        raise HTTPException(status_code=400, detail="backtest details not available. please run /run-strategy first.")
    return config.BACKTEST_RESULTS

@app.get("/training-metrics")
def training_metrics():
    # we return the training history or raise if none exist
    if config.TRAINING_HISTORY is None:
        raise HTTPException(status_code=400, detail="training metrics not available. please run /run-strategy first.")
    return config.TRAINING_HISTORY

@app.get("/performance-periods")
def performance_periods():
    # we analyze the stored equity_curve (from backtest) for performance segments
    if config.BACKTEST_RESULTS is None or "equity_curve" not in config.BACKTEST_RESULTS:
        raise HTTPException(status_code=400, detail="performance periods not available. please run /run-strategy first.")
    periods = analyze_equity_curve(config.BACKTEST_RESULTS["equity_curve"])
    return periods

@app.get("/training-analysis")
def training_analysis():
    # we analyze the training history to find best/worst epochs
    if config.TRAINING_HISTORY is None:
        raise HTTPException(status_code=400, detail="training analysis not available. please run /run-strategy first.")
    analysis = analyze_training_history(config.TRAINING_HISTORY)
    return analysis

if __name__ == "__main__":
    # we retrieve the port or default to 4000, then run the uvicorn server
    port = int(os.environ.get("PORT", 4000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)