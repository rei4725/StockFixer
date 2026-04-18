from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LightGBMModel(BaseModel):
    """
    LightGBMを用いた価格予測モデル。
    BaseModelを継承し、学習と予測のメソッドを実装する。
    """

    def __init__(self, model_name: str = "LightGBMModel", **kwargs):
        super().__init__(model_name)
        defaults: dict = {
            "n_estimators": 500,
            "max_depth": 4,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
        defaults.update(kwargs)
        self.model = lgb.LGBMRegressor(**defaults)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list | None = None,
    ):
        """
        LightGBMモデルを学習させる。
        Args:
            X (pd.DataFrame): 特徴量データ（学習用）。
            y (pd.Series): ターゲット変数（学習用）。
            eval_set: 検証データリスト。指定時は early stopping を有効化する。
                      例: [(X_val, y_val)]
        """
        logger.info(f"{self.model_name} の学習を開始します...")
        fit_kwargs: dict = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        self.model.fit(X, y, **fit_kwargs)
        logger.info(f"{self.model_name} の学習が完了しました。")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        LightGBMモデルで予測を行う。
        Args:
            X (pd.DataFrame): 予測対象の特徴量データ。
        Returns:
            pd.Series: 予測結果。
        """
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise ValueError("モデルが学習されていません。train()メソッドを実行してください。")
        logger.debug(f"{self.model_name} で予測を実行します...")
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)
