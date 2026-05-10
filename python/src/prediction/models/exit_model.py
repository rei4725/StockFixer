"""ML ベースエグジット最適化モデル (R-402)

保有後の最適売却タイミングを XGBoost 分類器で予測する。
特徴量: マルチホライズン予測スコア・信頼度等（予測 API から取得済みの列のみ使用）
ターゲット: 今日が最適エグジットタイムか否か（バイナリ分類）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.prediction.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ヒンドサイトラベリングパラメータ
_LABEL_LOOKAHEAD = 5   # N日後までの最高値と現在値を比較
_LABEL_PEAK_RATIO = 0.98  # 現在値が将来最高値の 98% 以上なら「今が最適」

# 予測 DataFrame から取得する特徴量列（nullable は 0 で補完）
FEATURE_COLS: list[str] = [
    "diff_ratio",
    "diff_ratio_3d",
    "diff_ratio_5d",
    "diff_ratio_10d",
    "confidence_ratio",
    "multi_horizon_score",
]


class ExitModel(BaseModel):
    """
    ML ベースエグジットタイミング予測モデル (R-402)。

    XGBoost 分類器で各保有日の「今が最適売却タイミングか」を予測する。
    predict() はエグジット確率 (0.0〜1.0) を返す。値が高いほど「今が最適売却」。
    """

    def __init__(self, model_name: str = "ExitModel", **kwargs: object) -> None:
        super().__init__(model_name)
        defaults: dict = {
            "n_estimators": 300,
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
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        defaults.update(kwargs)
        self.model = xgb.XGBClassifier(**defaults)

    # ------------------------------------------------------------------
    # BaseModel インターフェース
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list | None = None,
    ) -> None:
        """
        ExitModel を学習させる。

        Args:
            X: 特徴量 DataFrame（FEATURE_COLS 列を含む）
            y: ターゲット Series（1=今が最適エグジット, 0=保有継続）
            eval_set: early stopping 用検証データ [(X_val, y_val)]
        """
        logger.info("%s の学習を開始します (samples=%d)...", self.model_name, len(X))
        fit_kwargs: dict = {"verbose": False}
        if eval_set is not None:
            self.model.set_params(early_stopping_rounds=30)
            fit_kwargs["eval_set"] = eval_set
        self.model.fit(X, y, **fit_kwargs)
        logger.info("%s の学習が完了しました。", self.model_name)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        各行のエグジット確率を返す。

        Returns:
            エグジット確率 Series (0.0〜1.0)。高いほど「今が最適売却タイミング」。
        """
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise ValueError("モデルが学習されていません。train()メソッドを実行してください。")
        logger.debug(
            "%s でエグジット確率を予測します (rows=%d)...", self.model_name, len(X)
        )
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=X.index, name="exit_prob")

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------

    @staticmethod
    def make_labels(
        close: pd.Series,
        lookahead: int = _LABEL_LOOKAHEAD,
        peak_ratio: float = _LABEL_PEAK_RATIO,
    ) -> pd.Series:
        """
        ヒンドサイトラベリングでエグジットラベルを生成する。

        現在の終値が今後 ``lookahead`` 日間の最高値の ``peak_ratio`` 以上であれば
        「今が最適エグジット」(1)、そうでなければ「保有継続」(0) とラベル付けする。

        末尾 lookahead 行は判定不能のため NaN を返す（呼び出し側で dropna すること）。

        Args:
            close: 終値 Series（日次、インデックスは日付）
            lookahead: 比較する将来の日数
            peak_ratio: 現在値が将来最高値の何割以上なら最適と判断するか

        Returns:
            ラベル Series（1=exit, 0=hold）。末尾 lookahead 行は NaN。
        """
        close_num = pd.to_numeric(close, errors="coerce")
        future_max = close_num.shift(-1).rolling(lookahead, min_periods=1).max()
        label = (close_num >= future_max * peak_ratio).astype("Int64")
        label.iloc[-lookahead:] = pd.NA
        return label.rename("exit_signal")

    @staticmethod
    def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        予測 DataFrame から ExitModel 用の特徴量行列を作成する。

        FEATURE_COLS に含まれない列は 0.0 で補完する。

        Args:
            df: prediction_results 由来の DataFrame

        Returns:
            FEATURE_COLS 列のみを持つ float64 DataFrame
        """
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_COLS:
            result[col] = pd.to_numeric(df.get(col, pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        return result
