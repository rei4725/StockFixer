import joblib  # モデルの保存・ロード用
import lightgbm as lgb
import pandas as pd
from src.models.base_model import BaseModel


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
        print(f"{self.model_name} の学習を開始します...")
        self.model.fit(X, y)
        print(f"{self.model_name} の学習が完了しました。")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        LightGBMモデルで予測を行う。
        Args:
            X (pd.DataFrame): 予測対象の特徴量データ。
        Returns:
            pd.Series: 予測結果。
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。train()メソッドを実行してください。")
        print(f"{self.model_name} で予測を実行します...")
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)

    def save_model(self, path: str):
        """
        学習済みLightGBMモデルを保存する。
        Args:
            path (str): モデルの保存パス。
        """
        try:
            joblib.dump(self.model, path)
            print(f"{self.model_name} モデルを {path} に保存しました。")
        except Exception as e:
            print(f"{self.model_name} モデルの保存中にエラーが発生しました: {e}")

    def load_model(self, path: str):
        """
        保存されたLightGBMモデルをロードする。
        Args:
            path (str): モデルのロードパス。
        """
        try:
            self.model = joblib.load(path)
            print(f"{self.model_name} モデルを {path} からロードしました。")
        except Exception as e:
            print(f"{self.model_name} モデルのロード中にエラーが発生しました: {e}")
