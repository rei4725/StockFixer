import os
from typing import Dict, Type

import pandas as pd

from src.models.base_model import BaseModel
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel
from src.utils.data_path_utils import ensure_dir, get_models_dir, get_models_subdir
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    AIモデルの管理、学習、予測を行うクラス。
    複数のモデルタイプをサポートし、モデルの保存・ロードも管理する。
    """

    def __init__(self, model_dir: str = None):
        self.models: Dict[str, BaseModel] = {}
        self.model_dir = model_dir if model_dir else get_models_dir()
        ensure_dir(self.model_dir)
        self._registered_models: Dict[str, Type[BaseModel]] = {
            "XGBoostModel": XGBoostModel,
            "LightGBMModel": LightGBMModel,
        }

    def register_model_type(self, name: str, model_class: Type[BaseModel]):
        """
        新しいモデルタイプを登録する。
        Args:
            name (str): モデルタイプの名前。
            model_class (Type[BaseModel]): BaseModelを継承したモデルクラス。
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError("登録するモデルクラスはBaseModelを継承している必要があります。")
        self._registered_models[name] = model_class
        logger.debug(f"モデルタイプ '{name}' を登録しました。")

    def create_model(self, model_type: str, model_name: str, **kwargs) -> BaseModel:
        """
        指定されたタイプのAIモデルインスタンスを作成する。
        Args:
            model_type (str): 作成するモデルのタイプ名 (例: "XGBoostModel", "LightGBMModel")。
            model_name (str): モデルインスタンスの名前。
            **kwargs: モデルのコンストラクタに渡す追加引数。
        Returns:
            BaseModel: 作成されたモデルインスタンス。
        Raises:
            ValueError: 未登録のモデルタイプが指定された場合。
        """
        if model_type not in self._registered_models:
            raise ValueError(
                f"未登録のモデルタイプ: {model_type}. 登録済みのタイプ: {list(self._registered_models.keys())}"
            )

        model_class = self._registered_models[model_type]
        model_instance = model_class(model_name=model_name, **kwargs)
        self.models[model_name] = model_instance
        logger.debug(f"モデル '{model_name}' ({model_type}) を作成しました。")
        return model_instance

    def get_model(self, model_name: str) -> BaseModel:
        """
        指定された名前のモデルインスタンスを取得する。
        Args:
            model_name (str): 取得するモデルの名前。
        Returns:
            BaseModel: モデルインスタンス。
        Raises:
            ValueError: 指定された名前のモデルが見つからない場合。
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' が見つかりません。")
        return self.models[model_name]

    def train_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        market: str = None,
        symbol: str = None,
        auto_save: bool = True,
        **train_kwargs,
    ):
        """
        指定されたモデルを学習させる。
        Args:
            model_name (str): 学習させるモデルの名前。
            X (pd.DataFrame): 特徴量データ。
            y (pd.Series): ターゲット変数。
            market (str, optional): 市場名。
            symbol (str, optional): 銘柄コードやティッカー。
            auto_save (bool): Trueの場合、学習完了後に自動保存する（デフォルト: True）。
            **train_kwargs: model.train() に渡す追加引数（eval_set 等）。
        """
        model = self.get_model(model_name)
        model.train(X, y, **train_kwargs)
        if auto_save:
            self.save_model(model_name, market=market, symbol=symbol)

    def predict_with_model(self, model_name: str, X: pd.DataFrame) -> pd.Series:
        """
        指定されたモデルで予測を行う。
        Args:
            model_name (str): 予測を行うモデルの名前。
            X (pd.DataFrame): 予測対象の特徴量データ。
        Returns:
            pd.Series: 予測結果。
        """
        model = self.get_model(model_name)
        return model.predict(X)

    def save_model(self, model_name: str, market: str = None, symbol: str = None):
        """
        指定されたモデルを保存する。
        Args:
            model_name (str): 保存するモデルの名前。
            market (str, optional): 市場名。
            symbol (str, optional): 銘柄コードやティッカー。
        """
        model = self.get_model(model_name)
        if market and symbol:
            save_dir = get_models_subdir(market, symbol)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{model_name}.joblib")
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        model.save_model(model_path)

    def load_model(
        self, model_name: str, model_type: str = None, market: str = None, symbol: str = None
    ):
        """
        指定されたモデルをロードする。
        Args:
            model_name (str): ロードするモデルの名前。
            model_type (str, optional): ロードするモデルのタイプ。
            market (str, optional): 市場名。
            symbol (str, optional): 銘柄コードやティッカー。
        """
        if market and symbol:
            model_path = os.path.join(get_models_subdir(market, symbol), f"{model_name}.joblib")
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")

        if model_name in self.models:
            model_instance = self.models[model_name]
        else:
            if model_type is None:
                # 登録済みモデルタイプ名からモデルタイプを推測
                for registered_name in self._registered_models:
                    base_name = registered_name.replace("Model", "")
                    if base_name in model_name:
                        model_type = registered_name
                        break
                else:
                    raise ValueError(f"モデル '{model_name}' はまだ作成されていません。model_typeを指定してください。")
            model_instance = self.create_model(model_type, model_name)

        model_instance.load_model(model_path)
        self.models[model_name] = model_instance  # ロードしたモデルインスタンスを更新
        logger.debug(f"モデル '{model_name}' をロードしました。")
        return model_instance
