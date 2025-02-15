from pydantic import BaseModel
from typing import List

class StrategyResponse(BaseModel):
    # we define the fields expected in the response of the /run-strategy endpoint
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
    # we define the fields for the input to the /predict endpoint
    sequence: List[List[float]]

class PredictResponse(BaseModel):
    # we define the fields returned by the /predict endpoint
    predicted_close: float
    decision: str
