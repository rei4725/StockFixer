from enum import Enum

from src.prediction.models.lightgbm import LightGBMModel
from src.prediction.models.transformer_model import TransformerModel
from src.prediction.models.xgboost import XGBoostModel
from src.utils.ml_base import BaseModel


class ModelType(str, Enum):
    XGBOOST = "XGBoostModel"
    LIGHTGBM = "LightGBMModel"
    TRANSFORMER = "TransformerModel"
    EXIT = "ExitModel"


__all__ = [
    "BaseModel",
    "LightGBMModel",
    "ModelType",
    "TransformerModel",
    "XGBoostModel",
]
