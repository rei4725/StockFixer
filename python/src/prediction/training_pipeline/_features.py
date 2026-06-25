"""特徴量の読み込み・前処理・寄与分析。

銘柄別モデル学習パイプラインのうち、DB からの特徴量読み込み（並列安全）と
SHAP / permutation importance による特徴量寄与分析を担う。
"""

import re

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config.settings import (
    EARNINGS_MASK_WINDOW_DAYS,
    FEATURE_SELECTION_DROP_RATIO,
    FEATURE_SELECTION_MIN_FEATURES,
    PERMUTATION_IMPORTANCE_REPEATS,
)
from src.prediction.db import load_excluded_features, save_feature_selection, save_shap_values
from src.prediction.ports import get_market_data_port
from src.prediction.types import FeatureLoadResult
from src.utils.db import load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _apply_feature_exclusions(X: pd.DataFrame, market: str, symbol: str) -> pd.DataFrame:
    excluded = load_excluded_features(market, symbol)
    if not excluded:
        return X

    remaining = [col for col in X.columns if col not in excluded]
    if not remaining:
        logger.warning("特徴量除外後に列が空になるため元の特徴量を維持: %s/%s", market, symbol)
        return X

    logger.info("特徴量自動除外を適用: %s/%s %d列", market, symbol, len(excluded))
    return X[remaining].copy()


def _mask_earnings_rows(df: pd.DataFrame, market: str, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    work = df.copy()
    has_date_column = "date" in work.columns
    if has_date_column:
        date_index = pd.DatetimeIndex(pd.to_datetime(work["date"], errors="coerce"))
    elif isinstance(work.index, pd.DatetimeIndex):
        date_index = pd.DatetimeIndex(work.index)
    else:
        return work

    if pd.Series(date_index).isna().all():
        return work

    work.index = date_index
    _mds = get_market_data_port()
    earnings_dates = _mds.get_earnings_dates(market, symbol)
    work = _mds.add_earnings_flag(work, earnings_dates, lookaround_days=EARNINGS_MASK_WINDOW_DAYS)
    work = work[work["earnings_flag"] == 0].copy()
    if has_date_column:
        work["date"] = work.index
    return work.drop(columns=["earnings_flag"], errors="ignore")


def _extract_mean_abs_shap_values(shap_values, feature_count: int) -> np.ndarray:
    """SHAPの戻り値形式差分を吸収し、特徴量ごとの平均絶対寄与を返す。"""
    values = getattr(shap_values, "values", shap_values)

    if isinstance(values, list):
        if not values:
            raise ValueError("SHAP values list is empty")
        per_output = [_extract_mean_abs_shap_values(item, feature_count) for item in values]
        return np.mean(np.vstack(per_output), axis=0)

    arr = np.asarray(values)
    if arr.ndim == 0:
        raise ValueError("SHAP values are scalar")

    if arr.ndim == 1:
        if arr.shape[0] != feature_count:
            raise ValueError(
                f"SHAP feature length mismatch: expected {feature_count}, got {arr.shape[0]}"
            )
        return np.abs(arr)

    feature_axis = next(
        (axis for axis in range(arr.ndim - 1, -1, -1) if arr.shape[axis] == feature_count),
        None,
    )
    if feature_axis is None:
        raise ValueError(
            "Unable to locate feature axis in SHAP values "
            f"shape={arr.shape}, feature_count={feature_count}"
        )

    reduce_axes = tuple(axis for axis in range(arr.ndim) if axis != feature_axis)
    mean_abs = np.abs(arr).mean(axis=reduce_axes) if reduce_axes else np.abs(arr)
    mean_abs = np.asarray(mean_abs).reshape(-1)
    if mean_abs.shape[0] != feature_count:
        raise ValueError(
            f"SHAP summary length mismatch: expected {feature_count}, got {mean_abs.shape[0]}"
        )
    return mean_abs


def load_features_for_training(market: str, symbol: str, horizon: int = 1) -> FeatureLoadResult:
    """
    学習用の特徴量データをDBから読み込む（DB書き込みなし、並列安全）。

    Args:
        market: 市場名（例: "us", "jp"）
        symbol: 銘柄コード（例: "AAPL", "7203"）
        horizon: 予測ホライズン（営業日）。1=翌日（DBのy列を使用）, N>1=OHLCV再計算。

    Returns:
        FeatureLoadResult: status / X / y を持つ型付き結果
    """
    try:
        logger.info(f"[データ読み込み] {market}/{symbol} (horizon={horizon}d)")

        if horizon == 1:
            # 既存パス: stock_features の y 列（翌日変化率）を使用
            df = load_stock_features(market, symbol)
            if df is None or df.empty:
                return FeatureLoadResult(
                    status="skip", market=market, symbol=symbol, reason="データなし"
                )

            df = _mask_earnings_rows(df, market, symbol)

            exclude_cols = ["y", "market", "symbol", "date"]
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            X = df[feature_cols]
            y = df["y"]
        else:
            # 多ホライズンパス: market_data_raw から OHLCV を取得して再計算
            from src.utils.db import load_raw_ohlcv

            raw = load_raw_ohlcv(market, symbol)
            if raw is None or raw.empty:
                return FeatureLoadResult(
                    status="skip",
                    market=market,
                    symbol=symbol,
                    reason="OHLCVデータなし",
                )

            # load_raw_ohlcv はすでに先頭大文字列名（Open/High/Low/Close/Volume）で返す
            _mds = get_market_data_port()
            earnings_dates = _mds.get_earnings_dates(market, symbol)
            raw = _mds.add_earnings_flag(
                raw, earnings_dates, lookaround_days=EARNINGS_MASK_WINDOW_DAYS
            )
            df_feat = _mds.add_technical_indicators(raw)
            X, y = _mds.create_basic_lag_features(df_feat, target_horizon=horizon)

        X = _apply_feature_exclusions(X, market, symbol)

        # 特徴量名の正規化
        def normalize_col(col):
            return re.sub(r"[^0-9a-zA-Z_]", "_", str(col))

        X = X.copy()
        X.columns = [normalize_col(c) for c in X.columns]

        return FeatureLoadResult(status="success", market=market, symbol=symbol, X=X, y=y)
    except Exception as e:
        logger.error(f"[データ読み込みエラー] {market}/{symbol}: {e}", exc_info=True)
        return FeatureLoadResult(status="error", market=market, symbol=symbol, error=str(e))


def _build_feature_selection_frame(
    X: pd.DataFrame,
    importance_mean: np.ndarray,
    importance_std: np.ndarray,
    protected_features: set[str] | None = None,
) -> pd.DataFrame:
    protected_features = protected_features or set()
    selection_df = (
        pd.DataFrame(
            {
                "feature": X.columns.tolist(),
                "importance_mean": importance_mean,
                "importance_std": importance_std,
            }
        )
        .sort_values(["importance_mean", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selection_df["importance_rank"] = range(1, len(selection_df) + 1)

    max_exclusions = max(0, len(selection_df) - FEATURE_SELECTION_MIN_FEATURES)
    exclude_count = min(int(len(selection_df) * FEATURE_SELECTION_DROP_RATIO), max_exclusions)
    excluded = (
        set(selection_df.tail(exclude_count)["feature"].tolist()) if exclude_count > 0 else set()
    )
    selection_df["protected_by_shap"] = selection_df["feature"].isin(protected_features)
    selection_df["is_excluded"] = (
        selection_df["feature"].isin(excluded) & ~selection_df["protected_by_shap"]
    )
    return selection_df


def _compute_and_save_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    market: str,
    symbol: str,
    model_name: str,
    trained_at: str,
    protected_features: set[str] | None = None,
) -> pd.DataFrame:
    """Permutation Importance を計算し、次回学習用の除外候補を保存する。"""
    if X.empty or len(X.columns) <= FEATURE_SELECTION_MIN_FEATURES:
        return pd.DataFrame()

    sample_size = min(len(X), 200)
    X_eval = X.tail(sample_size)
    y_eval = y.reindex(X_eval.index)
    result = permutation_importance(
        model.model,
        X_eval,
        y_eval,
        n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
        random_state=42,
        scoring="neg_mean_squared_error",
    )
    selection_df = _build_feature_selection_frame(
        X_eval,
        importance_mean=result.importances_mean,
        importance_std=result.importances_std,
        protected_features=protected_features,
    )
    save_feature_selection(market, symbol, model_name, trained_at, selection_df)
    return selection_df


def _compute_and_save_shap(
    model,
    X: pd.DataFrame,
    market: str,
    symbol: str,
    model_name: str,
    trained_at: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    SHAP値を計算してDBに保存し、上位・下位特徴量のDataFrameを返す。

    Args:
        model: 学習済みモデルインスタンス（XGBoostModel / LightGBMModel）
        X: 特徴量行列
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_name: モデル名
        trained_at: 学習日時文字列
        top_n: Discord通知する上位・下位N件

    Returns:
        pd.DataFrame: [feature, shap_mean, shap_rank] の上位+下位N件
    """
    try:
        import shap

        # サンプルサイズが大きい場合は計算コスト削減のため最大500行を使用
        X_sample = X.iloc[-500:] if len(X) > 500 else X

        explainer = shap.TreeExplainer(model.model)
        shap_arr = explainer.shap_values(X_sample)

        # SHAP Explanation / list / ndarray の差分を吸収する
        mean_abs = _extract_mean_abs_shap_values(shap_arr, feature_count=len(X_sample.columns))
        shap_df = pd.DataFrame({"feature": X_sample.columns.tolist(), "shap_mean": mean_abs})
        shap_df = shap_df.sort_values("shap_mean", ascending=False).reset_index(drop=True)
        shap_df["shap_rank"] = range(1, len(shap_df) + 1)

        save_shap_values(market, symbol, model_name, trained_at, shap_df)

        # 上位N件 + 下位N件を返す
        top = shap_df.head(top_n)
        bottom = shap_df.tail(top_n)
        return pd.concat([top, bottom], ignore_index=True).drop_duplicates(subset=["feature"])
    except Exception as e:
        logger.warning(f"SHAP計算スキップ [{market}_{symbol}/{model_name}]: {e}", exc_info=True)
        return pd.DataFrame()
