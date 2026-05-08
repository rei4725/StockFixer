import hashlib
import os
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional, Type

import joblib
import pandas as pd

from src.prediction.models.base import BaseModel
from src.prediction.models.lightgbm import LightGBMModel
from src.prediction.models.xgboost import XGBoostModel
from src.utils.data_path_utils import ensure_dir, get_models_dir, get_models_subdir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_feature_hash(feature_columns: List[str]) -> str:
    """特徴量カラム順のMD5ハッシュを返す。"""
    return hashlib.md5(",".join(feature_columns).encode()).hexdigest()


def _get_git_sha() -> str:
    """現在の git commit SHA（short）を返す。取得できない場合は空文字列。"""
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


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
            self.save_model(
                model_name, market=market, symbol=symbol, feature_columns=list(X.columns)
            )

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

    def save_model(
        self,
        model_name: str,
        market: str = None,
        symbol: str = None,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        指定されたモデルをメタデータ付き dict としてjoblib形式で保存する。

        保存形式: {"model": <raw model>, "feature_hash": str|None, "git_sha": str, "trained_at": str}
        """
        model = self.get_model(model_name)
        if market and symbol:
            save_dir = get_models_subdir(market, symbol)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{model_name}.joblib")
        else:
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")

        artifact = {
            "model": model.model,
            "feature_hash": _compute_feature_hash(feature_columns)
            if feature_columns is not None
            else None,
            "git_sha": _get_git_sha(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(artifact, model_path)
        logger.info(f"{model_name} モデルを {model_path} に保存しました。")

    def load_model(
        self,
        model_name: str,
        model_type: str = None,
        market: str = None,
        symbol: str = None,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        指定されたモデルをロードする。

        新形式（dict artifact）と旧形式（rawモデル）の両方に対応。
        feature_columns が指定された場合、保存時の特徴量ハッシュと比較し不一致なら警告を出す。
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

        artifact = joblib.load(model_path)
        if isinstance(artifact, dict) and "model" in artifact:
            model_instance.model = artifact["model"]
            self._check_feature_hash(
                model_name=model_name,
                artifact=artifact,
                feature_columns=feature_columns,
            )
            logger.info(
                f"{model_name} モデルを {model_path} からロードしました。"
                f" git_sha={artifact.get('git_sha', '')}"
                f" trained_at={artifact.get('trained_at', '')}"
            )
        else:
            # 旧形式（後方互換）
            model_instance.model = artifact
            logger.info(f"{model_name} モデルを {model_path} から旧形式でロードしました。")

        self.models[model_name] = model_instance
        return model_instance

    def _check_feature_hash(
        self,
        model_name: str,
        artifact: dict,
        feature_columns: Optional[List[str]],
    ) -> None:
        """artifact の feature_hash と現在の特徴量を比較し、不一致なら警告する。"""
        stored_hash = artifact.get("feature_hash")
        if stored_hash is None or feature_columns is None:
            return

        current_hash = _compute_feature_hash(feature_columns)
        if current_hash == stored_hash:
            return

        msg = (
            f"[特徴量不一致] モデル '{model_name}': "
            f"学習時ハッシュ={stored_hash} / 現在ハッシュ={current_hash} "
            f"(git_sha={artifact.get('git_sha', 'unknown')},"
            f" trained_at={artifact.get('trained_at', 'unknown')})"
        )
        logger.warning(msg)
        try:
            from src.reporting.discord.discord_utils import send_webhook_notification

            send_webhook_notification(
                title="⚠️ 特徴量ハッシュ不一致",
                message=msg,
                color=0xFF6600,
            )
        except Exception as e:
            logger.error("Discord通知失敗（特徴量不一致警告）: %s", e, exc_info=True)
