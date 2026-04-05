"""
統合モデルパイプラインサービス

全銘柄のデータを結合して1つのモデルを学習するサービス
"""

import os
from datetime import datetime
from typing import Tuple

import pandas as pd
from sklearn.inspection import permutation_importance

from config.settings import (
    EARNINGS_MASK_WINDOW_DAYS,
    FEATURE_SELECTION_DROP_RATIO,
    FEATURE_SELECTION_MIN_FEATURES,
    PERMUTATION_IMPORTANCE_REPEATS,
)
from src.data.data_loader import get_earnings_dates
from src.features.technical_analysis import add_earnings_flag
from src.utils.data_path_utils import ensure_dir, get_models_dir
from src.utils.db import load_all_stock_features, load_excluded_features, save_feature_selection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _mask_earnings_rows_for_unified(df: pd.DataFrame) -> pd.DataFrame:
    if (
        df.empty
        or "date" not in df.columns
        or "market" not in df.columns
        or "symbol" not in df.columns
    ):
        return df

    masked_frames: list[pd.DataFrame] = []
    removed_count = 0
    for (market, symbol), group in df.groupby(["market", "symbol"], sort=False):
        work = group.copy()
        date_index = pd.DatetimeIndex(pd.to_datetime(work["date"], errors="coerce"))
        work.index = date_index
        earnings_dates = get_earnings_dates(str(market), str(symbol))
        work = add_earnings_flag(work, earnings_dates, lookaround_days=EARNINGS_MASK_WINDOW_DAYS)
        before = len(work)
        work = work[work["earnings_flag"] == 0].copy()
        removed_count += before - len(work)
        work["date"] = work.index
        masked_frames.append(work.drop(columns=["earnings_flag"], errors="ignore"))

    if removed_count > 0:
        logger.info(f"統合学習用に決算近傍データを除外: {removed_count}行")
    return pd.concat(masked_frames, ignore_index=True) if masked_frames else df.iloc[0:0].copy()


def load_all_stock_data(data_dir: str = None) -> pd.DataFrame:
    """
    全銘柄のデータをDBから読み込み、結合して返す

    Args:
        data_dir: 後方互換のため残置（未使用、DBから直接取得）

    Returns:
        pd.DataFrame: 全銘柄の結合データ（market, symbol列付き）
    """
    df = load_all_stock_features()
    if df.empty:
        return pd.DataFrame()

    # 銘柄数カウント
    if "market" in df.columns and "symbol" in df.columns:
        n_symbols = df.groupby(["market", "symbol"]).ngroups
        logger.info(f"全銘柄データ読み込み: {len(df)}行, {n_symbols}銘柄")

    return _mask_earnings_rows_for_unified(df)


def _apply_unified_feature_exclusions(X: pd.DataFrame) -> pd.DataFrame:
    excluded = load_excluded_features("unified", "all", require_all_models=False)
    if not excluded:
        return X

    remaining = [col for col in X.columns if col not in excluded]
    return X[remaining].copy() if remaining else X


def _compute_and_save_unified_feature_selection(model, X: pd.DataFrame, y: pd.Series) -> None:
    if X.empty or len(X.columns) <= FEATURE_SELECTION_MIN_FEATURES:
        return

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
    selection_df = (
        pd.DataFrame(
            {
                "feature": X_eval.columns.tolist(),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
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
    selection_df["is_excluded"] = selection_df["feature"].isin(excluded)
    selection_df["protected_by_shap"] = False
    save_feature_selection(
        "unified",
        "all",
        model.model_name,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        selection_df,
    )


def prepare_unified_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    統合モデル用の特徴量を準備する

    Args:
        df: 全銘柄の結合データ

    Returns:
        X: 特徴量DataFrame
        y: ターゲットSeries
    """
    # market_encoded列がない場合のみ作成（後方互換性）
    if "market_encoded" not in df.columns and "market" in df.columns:
        market_codes = {"us": 0, "jp": 1}
        df["market_encoded"] = df["market"].map(market_codes).fillna(-1).astype(int)

    # y列があるか確認
    if "y" in df.columns:
        y = df["y"]
        # 特徴量から除外する列（文字列列・日付列・ターゲット）
        exclude_cols = ["market", "symbol", "y", "date"]
        feature_cols = [
            c
            for c in df.columns
            if c not in exclude_cols and not pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        X = df[feature_cols]
    else:
        # y列がない場合は作成（翌日終値）
        raise ValueError("y列がデータに含まれていません。特徴量生成済みのCSVを使用してください。")

    # NaNを除去
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    X = _apply_unified_feature_exclusions(X)

    return X, y


def train_unified_model(
    model_type: str = "XGBoostModel",
    model_name: str = "UnifiedStockModel",
    data_dir: str = None,
    save_dir: str = None,
    horizon: int = 1,
) -> None:
    """
    全銘柄のデータを使って統合モデルを学習・保存する

    Args:
        model_type: モデルタイプ ("XGBoostModel" or "LightGBMModel")
        model_name: 保存するモデル名
        data_dir: データディレクトリ（未使用、後方互換）
        save_dir: モデル保存ディレクトリ
        horizon: 予測ホライズン（営業日）。1=翌日（デフォルト）。
    """
    from src.models.model_manager import ModelManager

    logger.info(f"=== 統合モデル学習開始 (horizon={horizon}d) ===")

    if horizon == 1:
        # 既存パス: stock_features の y 列（翌日変化率）を使用
        logger.info("1. データ読み込み中...")
        df = load_all_stock_data(data_dir)
        if df.empty:
            logger.warning("データが見つかりません。")
            return
        logger.info("2. 特徴量準備中...")
        X, y = prepare_unified_features(df)
    else:
        # 多ホライズンパス: market_data_raw から全銘柄 OHLCV を取得して再計算
        logger.info("1. OHLCV データ読み込み中（全銘柄）...")
        X, y = _load_all_ohlcv_for_unified(horizon)
        if X is None or X.empty:
            logger.warning("OHLCVデータが見つかりません。")
            return

    X = _apply_unified_feature_exclusions(X)

    logger.info(f"特徴量数: {len(X.columns)}, サンプル数: {len(X)}")

    # モデル作成・学習
    logger.info(f"3. モデル学習中 ({model_type})...")
    mm = ModelManager(model_dir=save_dir)
    model = mm.create_model(model_type, model_name)
    model.train(X, y)
    _compute_and_save_unified_feature_selection(model, X, y)

    # 保存（unified/ディレクトリに保存）
    logger.info("4. モデル保存中...")
    if save_dir is None:
        save_dir = os.path.join(get_models_dir(), "unified")
    ensure_dir(save_dir)

    save_path = os.path.join(save_dir, f"{model_name}.joblib")
    model.save_model(save_path)
    logger.info(f"保存完了: {save_path}")

    logger.info(f"=== 統合モデル学習完了 (horizon={horizon}d) ===")


def _load_all_ohlcv_for_unified(horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    market_data_raw から全銘柄の OHLCV を取得し、指定ホライズンの特徴量を生成する。

    Args:
        horizon: 予測ホライズン（営業日）

    Returns:
        (X, y): 特徴量DataFrame と ターゲットSeries
    """
    import re

    from src.features.technical_analysis import add_technical_indicators, create_basic_lag_features
    from src.utils.db import load_all_raw_ohlcv_symbols, load_raw_ohlcv

    all_X = []
    all_y = []

    symbols = load_all_raw_ohlcv_symbols()
    logger.info(f"OHLCV対象銘柄数: {len(symbols)}")

    for market, symbol in symbols:
        try:
            raw = load_raw_ohlcv(market, symbol)
            if raw is None or raw.empty:
                continue

            # load_raw_ohlcv はすでに先頭大文字列名（Open/High/Low/Close/Volume）で返す
            df_feat = add_technical_indicators(raw)
            X_sym, y_sym = create_basic_lag_features(df_feat, target_horizon=horizon)

            # 銘柄を識別する特徴量を追加
            market_codes = {"us": 0, "jp": 1}
            X_sym = X_sym.copy()
            X_sym["market_encoded"] = market_codes.get(market, -1)

            def _norm(c):
                return re.sub(r"[^0-9a-zA-Z_]", "_", str(c))

            X_sym.columns = [_norm(c) for c in X_sym.columns]

            all_X.append(X_sym)
            all_y.append(y_sym)
        except Exception as e:
            logger.warning(f"OHLCV特徴量生成スキップ [{market}/{symbol}]: {e}")

    if not all_X:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]


def load_unified_model(model_name: str = "UnifiedStockModel", model_dir: str = None):
    """
    統合モデルをロードする

    Args:
        model_name: モデル名
        model_dir: モデルディレクトリ

    Returns:
        ロードされたモデル
    """
    from src.models.model_manager import ModelManager

    if model_dir is None:
        model_dir = os.path.join(get_models_dir(), "unified")

    mm = ModelManager(model_dir=model_dir)

    # XGBoostかLightGBMかを判定
    xgb_path = os.path.join(model_dir, f"{model_name}.joblib")
    if os.path.exists(xgb_path):
        # モデルタイプを判定するためファイル名から推測
        if "LightGBM" in model_name:
            model = mm.create_model("LightGBMModel", model_name)
        else:
            model = mm.create_model("XGBoostModel", model_name)
        model.load_model(xgb_path)
        return model

    raise FileNotFoundError(f"統合モデルが見つかりません: {xgb_path}")
