from pydantic import BaseModel
from typing import List

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
    sequence: List[List[float]]

class PredictResponse(BaseModel):
    predicted_close: float
    decision: str
