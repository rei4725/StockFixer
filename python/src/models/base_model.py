from abc import ABC, abstractmethod
import pandas as pd

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
        学習済みモデルを保存するメソッド。
        Args:
            path (str): モデルの保存パス。
        """
        # TODO: モデル保存の実装 (例: joblib, pickle)
        print(f"モデルを {path} に保存します。")
        pass

    def load_model(self, path: str):
        """
        保存されたモデルをロードするメソッド。
        Args:
            path (str): モデルのロードパス。
        """
        # TODO: モデルロードの実装 (例: joblib, pickle)
        print(f"モデルを {path} からロードします。")
        pass

    def get_model_name(self) -> str:
        """
        モデル名を取得する。
        Returns:
            str: モデル名。
        """
        return self.model_name
