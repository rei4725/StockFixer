"""ML モデル基底クラス。BC 横断で利用可能な汎用ユーティリティ。"""
from abc import ABC, abstractmethod

import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    AIモデルのベースクラス。
    すべてのAIモデルはこのクラスを継承し、学習と予測のメソッドを実装する。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    def save_model(self, path: str):
        try:
            joblib.dump(self.model, path)
            logger.info(f"{self.model_name} モデルを {path} に保存しました。")
        except Exception as e:
            logger.error(f"{self.model_name} モデルの保存中にエラーが発生しました: {e}", exc_info=True)
            raise

    def load_model(self, path: str):
        try:
            self.model = joblib.load(path)
            logger.info(f"{self.model_name} モデルを {path} からロードしました。")
        except Exception as e:
            logger.error(f"{self.model_name} モデルのロード中にエラーが発生しました: {e}", exc_info=True)
            raise

    def get_model_name(self) -> str:
        return self.model_name
