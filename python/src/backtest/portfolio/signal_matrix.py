"""全銘柄の予測スコア × 終値マトリクスを構築する。

Issue #511: 肥大化した portfolio.py を責務分割。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.ports import get_model_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_signal_matrix(
    symbols: list[tuple[str, str]],
    model_type: str,
    train_ratio: float,
    source: str,
    threshold: float,
    ensemble: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    全銘柄について学習 → 予測し、date × symbol の
    スコアマトリクスと Close 価格マトリクスを返す。

    学習データは train_ratio で分割し、テスト期間の予測のみ収録する。
    """
    from src.backtest.pipeline import _ensemble_predict, load_features

    score_dict: dict[str, pd.Series] = {}
    close_dict: dict[str, pd.Series] = {}

    for market, symbol in symbols:
        key = f"{market}_{symbol}"
        try:
            df = load_features(market, symbol, source)
            if df is None or df.empty or len(df) < 30:
                logger.debug(f"[PF] データ不足スキップ: {key}")
                continue

            # 学習 / テスト分割
            split = int(len(df) * train_ratio)
            if split < 20 or (len(df) - split) < 5:
                logger.debug(f"[PF] 分割後データ不足スキップ: {key}")
                continue

            train_df = df.iloc[:split]
            test_df = df.iloc[split:]

            exclude = {"y", "market", "symbol", "market_encoded", "Close", "close"}
            feature_cols = [c for c in df.columns if c not in exclude]

            X_train = train_df[feature_cols].dropna()
            y_train = train_df.loc[X_train.index, "y"].dropna()
            X_train = X_train.loc[y_train.index]

            X_test = test_df[feature_cols].dropna()

            if X_train.empty or X_test.empty:
                logger.debug(f"[PF] 特徴量空スキップ: {key}")
                continue

            mm = get_model_manager()

            if ensemble:
                pred = _ensemble_predict(mm, X_train, y_train, X_test, f"PF_{key}")
            else:
                mm.create_model(model_type, f"PF_{key}")
                mm.train_model(f"PF_{key}", X_train, y_train)
                raw = mm.predict_with_model(f"PF_{key}", X_test)
                pred = pd.Series(raw, index=X_test.index)

            # threshold 以上の予測値のみシグナルとして扱う（0未満はマスク）
            pred_filtered = pred.where(pred >= threshold, other=np.nan)
            score_dict[key] = pred_filtered

            close_col = "Close" if "Close" in test_df.columns else "close"
            if close_col in test_df.columns:
                close_dict[key] = test_df[close_col].reindex(X_test.index)

            logger.info(f"[PF] 予測完了: {key} ({len(pred)}行)")

        except Exception as e:
            logger.error(f"[PF] 銘柄スキップ ({key}): {e}", exc_info=True)

    if not score_dict:
        return pd.DataFrame(), pd.DataFrame()

    score_matrix = pd.DataFrame(score_dict).sort_index()
    close_matrix = pd.DataFrame(close_dict).sort_index()

    # 共通日付に揃える
    common_idx = score_matrix.index.intersection(close_matrix.index)
    score_matrix = score_matrix.loc[common_idx]
    close_matrix = close_matrix.loc[common_idx]

    return score_matrix, close_matrix
