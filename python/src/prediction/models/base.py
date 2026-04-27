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
        """
        モデルを学習させる抽象メソッド。
        Args:
            X (pd.DataFrame): 特徴量データ。
            y (pd.Series): ターゲット変数。
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        予測を行う抽象メソッド。
        Args:
            X (pd.DataFrame): 予測対象の特徴量データ。
        Returns:
            pd.Series: 予測結果。
        """
        pass

    def save_model(self, path: str):
        """
        学習済みモデルを joblib 形式で保存する。
        Args:
            path (str): モデルの保存パス。
        """
        try:
            joblib.dump(self.model, path)
            logger.info(f"{self.model_name} モデルを {path} に保存しました。")
        except Exception as e:
            logger.error(f"{self.model_name} モデルの保存中にエラーが発生しました: {e}", exc_info=True)
            raise

    def load_model(self, path: str):
        """
        Joblib 形式のモデルをロードする。
        Args:
            path (str): モデルのロードパス。
        """
        try:
            self.model = joblib.load(path)
            logger.info(f"{self.model_name} モデルを {path} からロードしました。")
        except Exception as e:
            logger.error(f"{self.model_name} モデルのロード中にエラーが発生しました: {e}", exc_info=True)
            raise

    def get_model_name(self) -> str:
        """
        モデル名を取得する。
        Returns:
            str: モデル名。
        """
        return self.model_name
