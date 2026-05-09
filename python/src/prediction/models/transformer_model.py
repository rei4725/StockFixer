from __future__ import annotations

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.prediction.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransformerModel(BaseModel):
    """
    MLPRegressor ベースのニューラルネットワークモデル。
    XGBoost / LightGBM（木ベース）とのアンサンブル多様性を向上させる。
    特徴量スケーリング（StandardScaler）を Pipeline に内包するため保存・ロードが一貫する。
    """

    def __init__(self, model_name: str = "TransformerModel", **kwargs):
        super().__init__(model_name)
        mlp_defaults: dict = {
            "hidden_layer_sizes": (128, 64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": 42,
        }
        mlp_defaults.update(kwargs)
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(**mlp_defaults)),
            ]
        )

    def train(self, X: pd.DataFrame, y: pd.Series, eval_set: list | None = None):
        logger.info(f"{self.model_name} の学習を開始します...")
        self.model.fit(X, y)
        logger.info(f"{self.model_name} の学習が完了しました。")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise ValueError("モデルが学習されていません。train()メソッドを実行してください。")
        logger.debug(f"{self.model_name} で予測を実行します...")
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)
