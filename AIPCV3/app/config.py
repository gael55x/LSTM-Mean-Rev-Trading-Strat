import datetime

# Settings
SEQ_LENGTH = 60
FEATURES = ['Close', 'pctB', 'RSI', 'MACD', 'Signal_Line', 'Momentum', 'ATR']
TICKER = 'BTC-USD'
DATA_DAYS = 7 * 365  

# Global state variables
MODEL = None
SCALER = None
TRAINING_HISTORY = None
BACKTEST_RESULTS = None
