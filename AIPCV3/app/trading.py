import numpy as np
import pandas as pd

def simulate_trading(
    test_data: pd.DataFrame,
    predicted_close: np.ndarray,
    actual_close: np.ndarray,
    transaction_cost=0.0005,
    stop_loss_percent=0.1,
    take_profit_percent=0.2,
    initial_capital=500.0
):
    """
    a revised trading simulation that attempts to reduce data leakage by:
      1) using next-day (shifted) predictions rather than same-day actual close.
      2) using fixed rsi and 'predicted change' thresholds, not full-sample quantiles.
    """

    # we copy the dataframe so we don't mutate the original
    test_data = test_data.copy()

    # we store the predicted and actual close arrays in the dataframe
    test_data['Predicted_Close'] = predicted_close
    test_data['Actual_Close'] = actual_close

    # ---------------
    # 1) shift the predictions forward one day
    #    so we pretend that today's "predicted close" is only used for tomorrow's decision.
    # ---------------
    test_data['Predicted_Close_Shifted'] = test_data['Predicted_Close'].shift(1)
    # we also shift rsi by one day to mimic only having 'yesterday's' rsi
    test_data['RSI_Shifted'] = test_data['RSI'].shift(1)

    # ---------------
    # 2) re-compute 'predicted_change' using the shifted values
    #    we compare yesterday's predicted close to yesterday's actual close.
    # ---------------
    test_data['Predicted_Change_Shifted'] = (
        test_data['Predicted_Close_Shifted'] - test_data['Actual_Close'].shift(1)
    ) / test_data['Actual_Close'].shift(1)

    # we drop rows with any na in critical columns
    test_data.dropna(
        subset=['Predicted_Close_Shifted', 'RSI_Shifted', 'Predicted_Change_Shifted'],
        inplace=True
    )
    # we initialize a 'signal' column that will be 1 for buy, -1 for sell, 0 for hold
    test_data['Signal'] = 0

    # ---------------
    # 3) replace full-sample quantiles with fixed thresholds
    #    (in production, you'd likely measure these or do walk-forward.)
    # ---------------
    RSI_buy_level = 40
    RSI_sell_level = 60
    buy_threshold = 0.01
    sell_threshold = -0.01

    # we assign a buy signal if predicted change is above 1% and rsi is below 40
    test_data.loc[
        (test_data['Predicted_Change_Shifted'] > buy_threshold) &
        (test_data['RSI_Shifted'] < RSI_buy_level),
        'Signal'
    ] = 1

    # we assign a sell signal if predicted change is below -1% and rsi is above 60
    test_data.loc[
        (test_data['Predicted_Change_Shifted'] < sell_threshold) &
        (test_data['RSI_Shifted'] > RSI_sell_level),
        'Signal'
    ] = -1

    # we simulate trades in a loop
    positions = []
    cash = initial_capital
    holdings = 0
    portfolio_values = []
    entry_price = None

    for index, row in test_data.iterrows():
        price = row['Actual_Close']
        signal = row['Signal']

        # if we get a buy signal and we have cash, buy
        if signal == 1 and cash > 0:
            amount_to_buy = (cash * 0.5) * (1 - transaction_cost)
            holdings += amount_to_buy / price
            cash -= amount_to_buy
            entry_price = price
            positions.append({'Date': str(index), 'Position': 'Buy', 'Price': price})

        # if we get a sell signal and hold something, sell
        elif signal == -1 and holdings > 0:
            amount_to_sell = holdings * price * (1 - transaction_cost)
            cash += amount_to_sell
            holdings = 0
            entry_price = None
            positions.append({'Date': str(index), 'Position': 'Sell', 'Price': price})

        # we check stop-loss and take-profit conditions if we hold a position
        elif holdings > 0 and entry_price is not None:
            if price <= entry_price * (1 - stop_loss_percent):
                amount_to_sell = holdings * price * (1 - transaction_cost)
                cash += amount_to_sell
                holdings = 0
                positions.append({
                    'Date': str(index),
                    'Position': 'Stop Loss Sell',
                    'Price': price
                })
                entry_price = None
            elif price >= entry_price * (1 + take_profit_percent):
                amount_to_sell = holdings * price * (1 - transaction_cost)
                cash += amount_to_sell
                holdings = 0
                positions.append({
                    'Date': str(index),
                    'Position': 'Take Profit Sell',
                    'Price': price
                })
                entry_price = None

        # we compute the daily portfolio value
        total_value = cash + holdings * price
        portfolio_values.append(total_value)

    # we attach the portfolio values back to the dataframe
    test_data['Portfolio_Value'] = portfolio_values

    # we compute daily and cumulative returns
    test_data['Daily_Return'] = test_data['Portfolio_Value'].pct_change()
    test_data['Cumulative_Return'] = (1 + test_data['Daily_Return']).cumprod()

    # we compute total days for annualized return
    total_days = (test_data.index[-1] - test_data.index[0]).days or 1
    annualized_return = (test_data['Cumulative_Return'].iloc[-1]) ** (365 / total_days) - 1

    returns = test_data['Daily_Return'].dropna()
    # we check for zero standard deviation to avoid divide-by-zero
    if returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # we compute max drawdown
    rolling_max = test_data['Portfolio_Value'].cummax()
    drawdown = test_data['Portfolio_Value'] / rolling_max - 1
    max_drawdown = drawdown.min()

    # we compute total return as a percentage gain/loss on the initial capital
    total_return = (test_data['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital * 100

    # we build a metrics dictionary to summarize performance
    metrics = {
        "num_buy_signals": int((test_data['Signal'] == 1).sum()),
        "num_sell_signals": int((test_data['Signal'] == -1).sum()),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "mae": None,  # to be updated later in main.py
        "mse": None,
        "rmse": None,
        "positions": positions
    }

    # we create an equity curve list to store date and portfolio value
    equity_curve = [
        {"date": str(date.date()), "portfolio_value": float(value)}
        for date, value in zip(test_data.index, test_data["Portfolio_Value"])
    ]

    # we gather everything into a results structure
    backtest_results = {
        "metrics": metrics,
        "equity_curve": equity_curve
    }

    return backtest_results, test_data
