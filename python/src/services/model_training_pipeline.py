"""
銘柄別モデル学習パイプラインサービス

指定した銘柄のデータを使い、XGBoost・LightGBMモデルを学習・保存する
"""

import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config.settings import (
    EARNINGS_MASK_WINDOW_DAYS,
    FEATURE_SELECTION_DROP_RATIO,
    FEATURE_SELECTION_MIN_FEATURES,
    FEATURE_SELECTION_PROTECT_TOP_SHAP,
    PERMUTATION_IMPORTANCE_REPEATS,
)
from src.data.data_loader import get_earnings_dates
from src.domain.types import FeatureLoadResult, TrainingMetrics
from src.features.technical_analysis import add_earnings_flag
from src.models.model_manager import ModelManager
from src.utils.db import (
    load_excluded_features,
    load_stock_features,
    save_feature_selection,
    save_model_metrics,
    save_shap_values,
)
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
    earnings_dates = get_earnings_dates(market, symbol)
    work = add_earnings_flag(work, earnings_dates, lookaround_days=EARNINGS_MASK_WINDOW_DAYS)
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
            from src.features.technical_analysis import (
                add_technical_indicators,
                create_basic_lag_features,
            )
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
            earnings_dates = get_earnings_dates(market, symbol)
            raw = add_earnings_flag(raw, earnings_dates, lookaround_days=EARNINGS_MASK_WINDOW_DAYS)
            df_feat = add_technical_indicators(raw)
            X, y = create_basic_lag_features(df_feat, target_horizon=horizon)

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


def _compute_training_metrics(y_true: pd.Series, y_pred: pd.Series) -> TrainingMetrics:
    """学習データでの in-sample 評価指標を計算する（モデル品質監視用）。"""
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    directional_accuracy = float((np.sign(y_pred) == np.sign(y_true)).mean())
    return TrainingMetrics(
        rmse=rmse, directional_accuracy=directional_accuracy, n_samples=len(y_true)
    )


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
        logger.warning(f"SHAP計算スキップ [{market}_{symbol}/{model_name}]: {e}")
        return pd.DataFrame()


def train_models_for_symbol(market: str, symbol: str, horizon: int = 1) -> dict:
    """
    単一銘柄に対してXGBoost・LightGBMモデルを学習・保存する

    Args:
        market: 市場名（例: "us", "jp"）
        symbol: 銘柄コード（例: "AAPL", "7203"）
        horizon: 予測ホライズン（営業日）。1=翌日（デフォルト）。

    Returns:
        dict: {"market", "symbol", "status", ...}  (batch_runner.print_summary 互換)
    """
    try:
        logger.info(f"[モデル作成開始] {market}/{symbol} (horizon={horizon}d)")

        # DBから特徴量データを取得
        loaded = load_features_for_training(market, symbol, horizon=horizon)
        if not loaded.is_success:
            return {
                "market": loaded.market,
                "symbol": loaded.symbol,
                "status": loaded.status,
                "reason": loaded.reason,
                "error": loaded.error,
            }

        X, y = loaded.X, loaded.y

        # horizon > 1 の場合はモデル名にサフィックスを付与
        suffix = f"_{horizon}d" if horizon > 1 else ""

        # ModelManagerは各呼び出しで新規作成
        model_manager = ModelManager()
        trained_at = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_type, model_name in [
            ("XGBoostModel", f"StockXGBoostModel{suffix}"),
            ("LightGBMModel", f"StockLightGBMModel{suffix}"),
        ]:
            model_manager.create_model(model_type, model_name)
            model_manager.train_model(model_name, X, y, market=market, symbol=symbol)
            # 学習後in-sample精度計測・DB記録
            try:
                model = model_manager.get_model(model_name)
                y_pred = model.predict(X)
                metrics = _compute_training_metrics(y, y_pred)
                save_model_metrics(market, symbol, model_name, trained_at, metrics)
                logger.debug(
                    f"[精度記録] {market}/{symbol}/{model_name}: "
                    f"RMSE={metrics.rmse:.6f}, "
                    f"方向正解率={metrics.directional_accuracy:.2%}"
                )
            except Exception as e:
                logger.warning(f"精度指標保存スキップ [{market}_{symbol}/{model_name}]: {e}")
            # SHAP特徴量寄与の計算・保存・Discord通知
            try:
                model = model_manager.get_model(model_name)
                shap_top_bottom = _compute_and_save_shap(
                    model, X, market, symbol, model_name, trained_at
                )
                protected_features = set(
                    shap_top_bottom.nsmallest(FEATURE_SELECTION_PROTECT_TOP_SHAP, "shap_rank")[
                        "feature"
                    ].tolist()
                )
                if not shap_top_bottom.empty:
                    from src.api.discord_utils import send_shap_notification

                    send_shap_notification(market, symbol, model_name, shap_top_bottom)
                _compute_and_save_permutation_importance(
                    model,
                    X,
                    y,
                    market,
                    symbol,
                    model_name,
                    trained_at,
                    protected_features=protected_features,
                )
            except Exception as e:
                logger.warning(f"SHAP/特徴量選択スキップ [{market}_{symbol}/{model_name}]: {e}")

        logger.info(f"[モデル作成完了] {market}/{symbol} (horizon={horizon}d)")
        return {"market": market, "symbol": symbol, "status": "success"}
    except Exception as e:
        logger.error(f"[モデル作成エラー] {market}/{symbol}: {e}", exc_info=True)
        return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}


def train_models_for_symbol_task(task) -> dict:
    """
    バッチランナー用ラッパー（SymbolTask または dict を受け取る）

    Args:
        task: SymbolTask または {"market": str, "symbol": str, "horizon": int (省略可)}

    Returns:
        dict: train_models_for_symbolの戻り値  (batch_runner.print_summary 互換)
    """
    from src.domain.types import SymbolTask

    if isinstance(task, SymbolTask):
        return train_models_for_symbol(task.market, task.symbol, task.horizon)
    return train_models_for_symbol(task["market"], task["symbol"], task.get("horizon", 1))


def run_model_batch(horizon: int = 1):
    """
    ウォッチリストの全銘柄のモデルを作成する。

    フェーズ1: DB読み込み（並列） - DuckDB読み取りはスレッド並列で安全
    フェーズ2: モデル学習・保存（逐次） - ファイルI/Oの競合を回避

    Args:
        horizon: 予測ホライズン（営業日）。1=翌日（デフォルト）。
    """
    from src.services.batch_runner import load_target_symbols, print_summary, run_parallel

    # バッチ作成の並列数（CPU数に応じて調整）
    MAX_MODEL_WORKERS = 3

    def _load_features_task(task) -> FeatureLoadResult:
        """バッチランナー用: DB読み込みのみ（並列安全）"""
        from src.domain.types import SymbolTask

        if isinstance(task, SymbolTask):
            return load_features_for_training(task.market, task.symbol, horizon=task.horizon)
        return load_features_for_training(
            task["market"], task["symbol"], horizon=task.get("horizon", 1)
        )

    symbols = load_target_symbols()
    if not symbols:
        logger.warning("対象銘柄がありません。")
        return

    # horizon 情報をタスクに付与
    from src.domain.types import SymbolTask

    tasks = [
        (
            SymbolTask(market=s.market, symbol=s.symbol, horizon=horizon)
            if hasattr(s, "market")
            else {**s, "horizon": horizon}
        )
        for s in symbols
    ]

    # フェーズ1: データ読み込み（並列）
    load_results = run_parallel(
        func=_load_features_task,
        tasks=tasks,
        max_workers=MAX_MODEL_WORKERS,
        label="データ読み込み",
    )

    # フェーズ2: モデル学習・保存（逐次）
    suffix = f"_{horizon}d" if horizon > 1 else ""
    success_data = [r for r in load_results if r.is_success]
    logger.info(f"モデル学習開始（逐次）: 対象件数={len(success_data)} (horizon={horizon}d)")

    trained_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_results = []
    for i, r in enumerate(success_data, 1):
        market, symbol = r.market, r.symbol
        try:
            logger.info(f"[モデル作成開始] {market}/{symbol}")
            model_manager = ModelManager()
            for model_type, model_name in [
                ("XGBoostModel", f"StockXGBoostModel{suffix}"),
                ("LightGBMModel", f"StockLightGBMModel{suffix}"),
            ]:
                model_manager.create_model(model_type, model_name)
                model_manager.train_model(model_name, r.X, r.y, market=market, symbol=symbol)
                try:
                    model = model_manager.get_model(model_name)
                    y_pred = model.predict(r.X)
                    metrics = _compute_training_metrics(r.y, y_pred)
                    save_model_metrics(market, symbol, model_name, trained_at, metrics)
                except Exception as me:
                    logger.warning(f"精度指標保存スキップ [{market}_{symbol}/{model_name}]: {me}")
            logger.info(f"[モデル作成完了] {market}/{symbol}")
            train_results.append({"market": market, "symbol": symbol, "status": "success"})
        except Exception as e:
            logger.error(f"[モデル作成エラー] {market}/{symbol}: {e}", exc_info=True)
            train_results.append(
                {"market": market, "symbol": symbol, "status": "error", "error": str(e)}
            )
        if i % 50 == 0:
            logger.info(f"  ... {i}/{len(success_data)} 件完了")

    # 最終サマリー
    final_results = train_results.copy()
    final_results += [r for r in load_results if r.status in ("error", "skip")]
    print_summary("モデル作成", final_results)
