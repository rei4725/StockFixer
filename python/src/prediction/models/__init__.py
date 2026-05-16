from enum import Enum

from src.prediction.models.base import BaseModel
from src.prediction.models.exit_model import ExitModel
from src.prediction.models.lightgbm import LightGBMModel
from src.prediction.models.transformer_model import TransformerModel
from src.prediction.models.xgboost import XGBoostModel


class ModelType(str, Enum):
    XGBOOST = "XGBoostModel"
    LIGHTGBM = "LightGBMModel"
    TRANSFORMER = "TransformerModel"
    EXIT = "ExitModel"


__all__ = [
    "BaseModel",
    "ExitModel",
    "LightGBMModel",
    "ModelType",
    "TransformerModel",
    "XGBoostModel",
]
