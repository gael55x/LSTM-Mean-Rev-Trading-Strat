import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from app.config import FEATURES, TICKER, DATA_DAYS

def download_data(ticker=TICKER):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=DATA_DAYS)
    data = yf.download(
        tickers=ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='1d',
        progress=False
    )
    data.dropna(inplace=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if 'Close' not in data.columns and 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']
    
    if data.empty or 'Close' not in data.columns:
        raise ValueError("No data downloaded or the 'Close' column is missing.")
    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    if 'Close' in data.columns:
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
    
        data['STD'] = data['Close'].rolling(window=20).std()
        data['Upper_Band'] = data['MA20'] + (data['STD'] * 2.5)
        data['Lower_Band'] = data['MA20'] - (data['STD'] * 2.5)
    
        close = data['Close']
        lower = data['Lower_Band']
        upper = data['Upper_Band']
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
    else:
        raise ValueError("The 'Close' column is missing from the data.")
    return data

def scale_data(data: pd.DataFrame):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[FEATURES])
    return scaled_data, scaler

def create_sequences(data_array: np.ndarray, seq_length: int):
    X, y = [], []
    for i in range(seq_length, len(data_array)):
        X.append(data_array[i - seq_length:i])
        y.append(data_array[i, 0])  # 'Close' is assumed to be the first feature
    return np.array(X), np.array(y)
