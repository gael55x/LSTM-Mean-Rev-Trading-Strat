import numpy as np
import pandas as pd

def simulate_trading(test_data: pd.DataFrame, predicted_close: np.ndarray, actual_close: np.ndarray,
                     transaction_cost=0.0005, stop_loss_percent=0.1, take_profit_percent=0.2,
                     initial_capital=500.0):
    test_data = test_data.copy()
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

    positions = []
    cash = initial_capital
    holdings = 0
    portfolio_value = []
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

    total_days = (test_data.index[-1] - test_data.index[0]).days or 1
    annualized_return = (test_data['Cumulative_Return'].iloc[-1]) ** (365 / total_days) - 1
    returns = test_data['Daily_Return'].dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0
    rolling_max = test_data['Portfolio_Value'].cummax()
    drawdown = test_data['Portfolio_Value'] / rolling_max - 1
    max_drawdown = drawdown.min()
    total_return = ((test_data['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital) * 100

    metrics = {
        "num_buy_signals": int((test_data['Signal'] == 1).sum()),
        "num_sell_signals": int((test_data['Signal'] == -1).sum()),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "mae": None,  # to be updated later
        "mse": None,
        "rmse": None,
        "positions": positions
    }

    equity_curve = [{"date": str(date.date()), "portfolio_value": value}
                    for date, value in zip(test_data.index, test_data["Portfolio_Value"])]

    backtest_results = {
        "metrics": metrics,
        "equity_curve": equity_curve
    }

    return backtest_results, test_data
