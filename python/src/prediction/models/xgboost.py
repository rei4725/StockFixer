from __future__ import annotations

import pandas as pd
import xgboost as xgb
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.prediction.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoostを用いた価格予測モデル。
    BaseModelを継承し、学習と予測のメソッドを実装する。
    """

    def __init__(self, model_name: str = "XGBoostModel", **kwargs):
        super().__init__(model_name)
        defaults: dict = {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
        }
        defaults.update(kwargs)
        self.model = xgb.XGBRegressor(**defaults)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list | None = None,
    ):
        """
        XGBoostモデルを学習させる。
        Args:
            X (pd.DataFrame): 特徴量データ（学習用）。
            y (pd.Series): ターゲット変数（学習用）。
            eval_set: 検証データリスト。指定時は early stopping を有効化する。
                      例: [(X_val, y_val)]
        """
        logger.info(f"{self.model_name} の学習を開始します...")
        fit_kwargs: dict = {"verbose": False}
        if eval_set is not None:
            self.model.set_params(early_stopping_rounds=50)
            fit_kwargs["eval_set"] = eval_set
        self.model.fit(X, y, **fit_kwargs)
        logger.info(f"{self.model_name} の学習が完了しました。")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        XGBoostモデルで予測を行う。
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
