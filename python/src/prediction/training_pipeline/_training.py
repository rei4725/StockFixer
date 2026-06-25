"""銘柄別モデル学習のオーケストレーション。

特徴量読み込み（_features）の結果を用いて XGBoost / LightGBM（任意で Transformer）
モデルを学習・保存し、in-sample / OOS 指標・SHAP 寄与・実験ランを記録する。
バッチ学習（並列読み込み + 逐次学習）と複数ホライズン学習もここに集約する。
"""

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.types import BatchResult

import numpy as np
import pandas as pd

from config.settings import FEATURE_SELECTION_PROTECT_TOP_SHAP
from src.prediction.db import save_model_metrics
from src.prediction.manager import ModelManager
from src.prediction.purged_cv import PurgedKFold
from src.prediction.types import FeatureLoadResult, TrainingMetrics
from src.utils.db import generate_run_id, save_experiment_run
from src.utils.logger import get_logger

from ._features import (
    _compute_and_save_permutation_importance,
    _compute_and_save_shap,
    load_features_for_training,
)

logger = get_logger(__name__)

# Purged K-Fold CV（#372）: leak-free な OOS 指標を算出する最小サンプル数とフォールド数。
# これ未満のデータでは従来のホールドアウト指標にフォールバックする。
_PURGED_CV_MIN_SAMPLES = 150
_PURGED_CV_N_SPLITS = 5
# CV はフォールドごとに一時モデルを学習するため、学習が軽量なモデルに限定する
# （TransformerModel は対象外 — ホールドアウト指標を使用）。
_PURGED_CV_MODEL_TYPES = {"XGBoostModel", "LightGBMModel"}


def _compute_training_metrics(y_true: pd.Series, y_pred: pd.Series) -> TrainingMetrics:
    """学習データでの in-sample 評価指標を計算する（モデル品質監視用）。"""
    rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
    directional_accuracy = float((np.sign(y_pred) == np.sign(y_true)).mean())
    return TrainingMetrics(
        rmse=rmse, directional_accuracy=directional_accuracy, n_samples=len(y_true)
    )


def _compute_purged_cv_metrics(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int,
) -> TrainingMetrics | None:
    """Purged K-Fold + Embargo CV で leak-free な OOS 指標を算出する（#372）。

    各フォールドで一時モデル（auto_save=False）を学習し、全フォールドの OOS 予測を
    プールして指標を計算する。前向きラベル（horizon 日先リターン）がテスト期間と
    重なる訓練サンプルは purge_gap=horizon で除外される。

    Returns:
        プール済み OOS 指標。データ不足・対象外モデル・学習失敗時は None
        （呼び出し元がホールドアウト指標へフォールバックする）。
    """
    if model_type not in _PURGED_CV_MODEL_TYPES:
        return None
    if len(X) < _PURGED_CV_MIN_SAMPLES:
        return None

    try:
        splitter = PurgedKFold(
            n_splits=_PURGED_CV_N_SPLITS,
            purge_gap=max(int(horizon), 1),
        )
        manager = ModelManager()
        y_true_parts: list[pd.Series] = []
        y_pred_parts: list[np.ndarray] = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
            fold_name = f"_purged_cv_{model_type}_{fold}"
            manager.create_model(model_type, fold_name)
            manager.train_model(fold_name, X.iloc[train_idx], y.iloc[train_idx], auto_save=False)
            pred = manager.get_model(fold_name).predict(X.iloc[test_idx])
            y_pred_parts.append(np.asarray(pred, dtype=float))
            y_true_parts.append(y.iloc[test_idx])

        if not y_true_parts:
            return None
        y_true = pd.concat(y_true_parts)
        y_pred = pd.Series(np.concatenate(y_pred_parts), index=y_true.index)
        return _compute_training_metrics(y_true, y_pred)
    except Exception as e:
        logger.warning(
            f"Purged CV 指標の計算に失敗（ホールドアウト指標にフォールバック）: {e}",
            exc_info=True,
        )
        return None


def train_models_for_symbol(
    market: str,
    symbol: str,
    horizon: int = 1,
    shadow_mode: bool = False,
    use_transformer: bool = False,
) -> dict:
    """
    単一銘柄に対してXGBoost・LightGBMモデルを学習・保存する

    shadow_mode=True の場合はチャレンジャーモデル（Challenger* 名）として保存する。
    本番モデル（Stock*）とは別ファイルに保存され、evaluate_shadow_models() で比較可能。

    Args:
        market: 市場名（例: "us", "jp"）
        symbol: 銘柄コード（例: "AAPL", "7203"）
        horizon: 予測ホライズン（営業日）。1=翌日（デフォルト）。
        shadow_mode: True のときチャレンジャーモデルとして学習・保存する
        use_transformer: True のとき TransformerModel も学習・保存する

    Returns:
        dict: {"market", "symbol", "status", ...}  (batch_runner.print_summary 互換)
    """
    try:
        mode_label = "challenger" if shadow_mode else "production"
        logger.info(f"[モデル作成開始] {market}/{symbol} (horizon={horizon}d, mode={mode_label})")

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

        # 時系列順に 80/20 分割（バリデーション: 直近 20%）
        # Purge（#372）: ラベルは horizon 営業日先のリターンのため、境界直前の
        # horizon 行は前向きラベルが検証期間と重なる（リーク）。学習集合から除外する。
        n_total = len(X)
        n_val = max(int(n_total * 0.2), min(30, n_total // 3))
        purge_gap = max(int(horizon), 1)
        if n_total - n_val - purge_gap >= 100:
            X_train, y_train = X.iloc[: -(n_val + purge_gap)], y.iloc[: -(n_val + purge_gap)]
            X_val, y_val = X.iloc[-n_val:], y.iloc[-n_val:]
            train_extra: dict = {"eval_set": [(X_val, y_val)]}
        else:
            # データ不足時は分割なしで学習（正則化のみ有効）
            X_train, y_train = X, y
            X_val, y_val = X, y
            train_extra = {}

        # ModelManagerは各呼び出しで新規作成
        model_manager = ModelManager()
        trained_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        shap_results: list[dict] = []

        # shadow_mode のときはチャレンジャーモデル名プレフィックスを使用
        name_prefix = "Challenger" if shadow_mode else "Stock"

        model_specs = [
            ("XGBoostModel", f"{name_prefix}XGBoostModel{suffix}"),
            ("LightGBMModel", f"{name_prefix}LightGBMModel{suffix}"),
        ]
        if use_transformer:
            model_specs.append(("TransformerModel", f"{name_prefix}TransformerModel{suffix}"))

        for model_type, model_name in model_specs:
            run_id = generate_run_id()
            model_manager.create_model(model_type, model_name)
            model_manager.train_model(
                model_name, X_train, y_train, market=market, symbol=symbol, **train_extra
            )
            # 学習後 out-of-sample 精度計測・DB記録
            # Purged K-Fold CV（#372）による leak-free 指標を優先し、
            # 算出不能時は従来のホールドアウト指標にフォールバックする。
            saved_metrics = None
            try:
                model = model_manager.get_model(model_name)
                saved_metrics = _compute_purged_cv_metrics(model_type, X, y, horizon)
                metrics_source = "purged-cv"
                if saved_metrics is None:
                    y_pred = model.predict(X_val)
                    saved_metrics = _compute_training_metrics(y_val, y_pred)
                    metrics_source = "holdout"
                save_model_metrics(market, symbol, model_name, trained_at, saved_metrics)
                logger.debug(
                    f"[精度記録] {market}/{symbol}/{model_name}: "
                    f"RMSE={saved_metrics.rmse:.6f}, "
                    f"方向正解率={saved_metrics.directional_accuracy:.2%} "
                    f"(OOS, {metrics_source})"
                )
            except Exception as e:
                logger.warning(
                    f"精度指標保存スキップ [{market}_{symbol}/{model_name}]: {e}", exc_info=True
                )
            # 実験ランを experiment_runs テーブルへ記録
            try:
                save_experiment_run(
                    run_id=run_id,
                    market=market,
                    symbol=symbol,
                    model_name=model_name,
                    trained_at=trained_at,
                    horizon=horizon,
                    rmse=saved_metrics.rmse if saved_metrics else None,
                    directional_accuracy=(
                        saved_metrics.directional_accuracy if saved_metrics else None
                    ),
                    n_samples=saved_metrics.n_samples if saved_metrics else None,
                    feature_names=list(X.columns),
                    params={"role": mode_label},
                )
            except Exception as e:
                logger.warning(
                    f"実験ラン保存スキップ [{market}_{symbol}/{model_name}]: {e}", exc_info=True
                )
            # SHAP特徴量寄与の計算・保存
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
                    shap_results.append(
                        {
                            "market": market,
                            "symbol": symbol,
                            "model_name": model_name,
                            "shap_top_bottom": shap_top_bottom,
                        }
                    )
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
                logger.warning(
                    f"SHAP/特徴量選択スキップ [{market}_{symbol}/{model_name}]: {e}", exc_info=True
                )

        logger.info(f"[モデル作成完了] {market}/{symbol} (horizon={horizon}d)")
        return {
            "market": market,
            "symbol": symbol,
            "status": "success",
            "shap_results": shap_results,
        }
    except Exception as e:
        logger.error(f"[モデル作成エラー] {market}/{symbol}: {e}", exc_info=True)
        return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}


def train_models_for_symbol_task(
    task, shadow_mode: bool = False, use_transformer: bool = False
) -> dict:
    """
    バッチランナー用ラッパー（SymbolTask または dict を受け取る）

    Args:
        task: SymbolTask または {"market": str, "symbol": str, "horizon": int (省略可)}
        shadow_mode: True のときチャレンジャーモデルとして学習・保存する
        use_transformer: True のとき TransformerModel も学習・保存する

    Returns:
        dict: train_models_for_symbolの戻り値  (batch_runner.print_summary 互換)
    """
    from src.domain.types import SymbolTask

    if isinstance(task, SymbolTask):
        return train_models_for_symbol(
            task.market,
            task.symbol,
            task.horizon,
            shadow_mode=shadow_mode,
            use_transformer=use_transformer,
        )
    return train_models_for_symbol(
        task["market"],
        task["symbol"],
        task.get("horizon", 1),
        shadow_mode=shadow_mode,
        use_transformer=use_transformer,
    )


def _log_training_summary(phase: str, batch_result: "BatchResult") -> None:
    """バッチ学習結果をログに出力する。"""
    n_success = len(batch_result.succeeded)
    n_skip = len(batch_result.skipped)
    n_error = len(batch_result.failed)
    logger.info(f"{phase} 結果サマリー 成功: {n_success} / スキップ: {n_skip} / エラー: {n_error}")
    if batch_result.failed:
        logger.warning(f"\n{phase} エラー詳細:")
        for f in batch_result.failed:
            logger.warning(f"  - {f.market}/{f.symbol}: {f.error}")


def run_batch_training(tasks: list, horizon: int = 1) -> "BatchResult":
    """
    ウォッチリストの全銘柄のモデルを作成する。

    tasks は呼び出し元（orchestration 等）で load_target_symbols() して渡す。

    フェーズ1: DB読み込み（並列） - DuckDB読み取りはスレッド並列で安全
    フェーズ2: モデル学習・保存（逐次） - ファイルI/Oの競合を回避

    Args:
        tasks: 対象銘柄タスクリスト（SymbolTask または dict）
        horizon: 予測ホライズン（営業日）。1=翌日（デフォルト）。
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeoutError
    from concurrent.futures import as_completed

    from src.domain.types import BatchFailure, BatchResult, SymbolTask

    _MAX_MODEL_WORKERS = 3
    _TASK_TIMEOUT = 300

    def _load_features_task(task) -> FeatureLoadResult:
        """DB読み込みのみ（並列安全）"""
        if isinstance(task, SymbolTask):
            return load_features_for_training(task.market, task.symbol, horizon=task.horizon)
        return load_features_for_training(
            task["market"], task["symbol"], horizon=task.get("horizon", 1)
        )

    if not tasks:
        logger.warning("対象銘柄がありません。")
        return BatchResult(succeeded=[], failed=[], skipped=[])

    # horizon 情報をタスクに付与
    enhanced_tasks = [
        (
            SymbolTask(market=s.market, symbol=s.symbol, horizon=horizon)
            if hasattr(s, "market")
            else {**s, "horizon": horizon}
        )
        for s in tasks
    ]

    # フェーズ1: データ読み込み（並列）
    logger.info(
        f"データ読み込み開始（並列数: {_MAX_MODEL_WORKERS}） 対象件数: {len(enhanced_tasks)}"
    )
    load_succeeded: list = []
    load_failed: list = []
    load_skipped: list = []

    with ThreadPoolExecutor(max_workers=_MAX_MODEL_WORKERS) as executor:
        futures = {executor.submit(_load_features_task, task): task for task in enhanced_tasks}
        for future in as_completed(futures):
            task = futures[future]
            _m = getattr(task, "market", None) or (
                task.get("market", "?") if hasattr(task, "get") else "?"
            )
            _s = getattr(task, "symbol", None) or (
                task.get("symbol", "?") if hasattr(task, "get") else "?"
            )
            try:
                result = future.result(timeout=_TASK_TIMEOUT)
                if result.status == "skip":
                    load_skipped.append(result)
                elif result.status == "error":
                    logger.error(f"[データ読み込みエラー] {_m}/{_s}: {result.error}")
                    load_failed.append(
                        BatchFailure(market=_m, symbol=_s, error=result.error or "読み込みエラー")
                    )
                else:
                    load_succeeded.append(result)
            except FuturesTimeoutError:
                logger.error(f"[タイムアウト] {_m}/{_s}: {_TASK_TIMEOUT}秒超過")
                load_failed.append(
                    BatchFailure(market=_m, symbol=_s, error=f"タイムアウト（{_TASK_TIMEOUT}秒）")
                )
            except Exception as e:
                logger.error(f"[未処理エラー] {_m}/{_s}: {e}", exc_info=True)
                load_failed.append(BatchFailure(market=_m, symbol=_s, error=str(e)))

    # フェーズ2: モデル学習・保存（逐次）
    suffix = f"_{horizon}d" if horizon > 1 else ""
    logger.info(f"モデル学習開始（逐次）: 対象件数={len(load_succeeded)} (horizon={horizon}d)")

    trained_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_results = []
    for i, r in enumerate(load_succeeded, 1):
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
                    logger.warning(
                        f"精度指標保存スキップ [{market}_{symbol}/{model_name}]: {me}",
                        exc_info=True,
                    )
            logger.info(f"[モデル作成完了] {market}/{symbol}")
            train_results.append({"market": market, "symbol": symbol, "status": "success"})
        except Exception as e:
            logger.error(f"[モデル作成エラー] {market}/{symbol}: {e}", exc_info=True)
            train_results.append(
                {"market": market, "symbol": symbol, "status": "error", "error": str(e)}
            )
        if i % 50 == 0:
            logger.info(f"  ... {i}/{len(load_succeeded)} 件完了")

    # 最終サマリー
    train_succeeded = [r for r in train_results if r.get("status") == "success"]
    train_failed = [
        BatchFailure(
            market=r.get("market", "?"),
            symbol=r.get("symbol", "?"),
            error=r.get("error", "モデル学習エラー"),
        )
        for r in train_results
        if r.get("status") == "error"
    ]
    final_batch = BatchResult(
        succeeded=train_succeeded,
        failed=train_failed + load_failed,
        skipped=load_skipped,
    )
    _log_training_summary("モデル作成", final_batch)
    return final_batch


def _train_models_for_horizon(horizon: int, tasks: list, max_workers: int = 3) -> list:
    """
    指定ホライズンの全銘柄モデルを学習する（機能テスト対応シンプル実装）。

    Args:
        horizon: 予測ホライズン（営業日）
        tasks: 対象銘柄タスクリスト（orchestration 側で load_target_symbols() して渡す）
        max_workers: 並列ワーカー数（現在は逐次処理）

    Returns:
        list[dict]: 各銘柄の学習結果サマリー
    """
    if not tasks:
        logger.warning("対象銘柄がありません。")
        return []

    results = []
    for sym in tasks:
        try:
            loaded = load_features_for_training(sym.market, sym.symbol, horizon=horizon)
            if not loaded.is_success:
                logger.debug(f"[学習スキップ] {sym.market}/{sym.symbol}: {loaded.status}")
                results.append(
                    {"market": sym.market, "symbol": sym.symbol, "status": loaded.status}
                )
                continue
            result = train_models_for_symbol(sym.market, sym.symbol, horizon=horizon)
            results.append(result)
        except Exception as e:
            logger.error(f"[学習エラー] {sym.market}/{sym.symbol}: {e}", exc_info=True)
            results.append(
                {
                    "market": sym.market,
                    "symbol": sym.symbol,
                    "status": "error",
                    "error": str(e),
                }
            )

    return results


def train_all_models(
    horizons: list | None = None, tasks: list | None = None, max_workers: int = 3
) -> None:
    """
    複数ホライズンで全銘柄のモデルを学習する。

    Args:
        horizons: 学習対象ホライズンのリスト（例: [1, 3, 7]）。None のとき [1] を使用。
        tasks: 対象銘柄タスクリスト。None のとき空リストとして扱う。
        max_workers: 並列ワーカー数
    """
    if horizons is None:
        horizons = [1]
    if tasks is None:
        tasks = []

    for horizon in horizons:
        try:
            logger.info(f"[全銘柄学習開始] horizon={horizon}d")
            _train_models_for_horizon(horizon, tasks, max_workers=max_workers)
        except Exception as e:
            logger.error(f"[ホライズン学習エラー] horizon={horizon}: {e}", exc_info=True)
