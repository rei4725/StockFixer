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
        self.model = lgb.LGBMRegressor(**kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        LightGBMモデルを学習させる。
        Args:
            X (pd.DataFrame): 特徴量データ。
            y (pd.Series): ターゲット変数。
        """
        logger.info(f"{self.model_name} の学習を開始します...")
        self.model.fit(X, y)
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
