"""ウォッチリスト全銘柄へのバッチ最適化（グリッド / Optuna）。

フェーズ1で並列に最適化計算（DuckDB 読み込みのみ）、
フェーズ2で逐次に結果保存（optimal_params.json の同時書き込み破損を防止）。
"""

from datetime import datetime, timedelta
from typing import Any, Optional

from src.backtest.optimizer.grid import run_optimization
from src.backtest.optimizer.optuna_search import run_optuna_optimization
from src.backtest.optimizer.persistence import (
    get_optimal_params,
    save_optimal_params_json,
    save_optimization_results,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_optimize_batch(
    tasks: list,
    model_type: str = "XGBoostModel",
    ensemble: bool = False,
    source: str = "file",
    n_splits: int = 5,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: Optional[float] = None,
    dynamic_slippage: bool = True,
    position_sizing: str = "full",
    position_fraction: float = 0.5,
    atr_risk_pcts: Optional[list[float]] = None,
    atr_multipliers: Optional[list[float]] = None,
    atr_min_fraction: float = 0.1,
    atr_max_fraction: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 0.015,
    threshold_step: float = 0.001,
    optimize_risk: bool = False,
    max_workers: int = 3,
    sort_by: str = "sharpe_ratio",
    skip_days: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    ウォッチリストの全銘柄に対してバックテスト最適化をバッチ実行する。

    tasks は呼び出し元（orchestration 等）で load_target_symbols() して渡す。

    フェーズ1: 最適化計算（並列） — DuckDB読み込みのみ・書き込みなし
    フェーズ2: 結果保存（逐次） — optimal_params.json の同時書き込み破損を防止

    Args:
        tasks: 対象銘柄タスクリスト（SymbolTask または dict）
        model_type: モデルタイプ
        ensemble: XGBoost+LightGBMアンサンブル予測を使用するか
        source: データソース
        n_splits: Walk-Forward分割数
        initial_cash: 初期資金
        fee_rate: 取引手数料率
        slippage: スリッページ
        position_sizing: ポジションサイジング種別
        position_fraction: fixed モード用比率
        atr_risk_pcts: ATR リスク割合の候補一覧
        atr_multipliers: ATR 倍率の候補一覧
        atr_min_fraction: ATR 建玉下限比率
        atr_max_fraction: ATR 建玉上限比率
        threshold_min: 閾値の最小値
        threshold_max: 閾値の最大値
        threshold_step: 閾値のステップ
        optimize_risk: ストップロス・テイクプロフィットもグリッドサーチするか
        max_workers: 並列数
        sort_by: 最適パラメータ選定基準
        skip_days: 最終最適化からこの日数以内ならスキップ（None=常に実行）

    Returns:
        各銘柄の結果サマリー list[dict]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    symbols = tasks
    if not symbols:
        logger.warning("対象銘柄がありません。")
        return []

    logger.info(f"全銘柄最適化バッチ開始: {len(symbols)}銘柄 / 並列数={max_workers}")

    def _optimize_task(task: Any) -> dict[str, Any]:
        market = getattr(task, "market", None) or task["market"]
        symbol = getattr(task, "symbol", None) or task["symbol"]

        # スキップ判定: 直近 skip_days 日以内に最適化済みならスキップ
        if skip_days is not None:
            existing = get_optimal_params(market, symbol)
            if existing and existing.get("timestamp"):
                try:
                    last_optimized = datetime.fromisoformat(existing["timestamp"])
                    if datetime.now() - last_optimized < timedelta(days=skip_days):
                        logger.info(
                            f"[{market}/{symbol}] スキップ"
                            f"（最終最適化: {last_optimized.strftime('%Y-%m-%d %H:%M')}"
                            f", {skip_days}日以内）"
                        )
                        return {"market": market, "symbol": symbol, "status": "skipped"}
                except ValueError:
                    pass  # timestamp パース失敗時は通常実行

        try:
            result_df = run_optimization(
                market=market,
                symbol=symbol,
                model_type=model_type,
                ensemble=ensemble,
                source=source,
                n_splits=n_splits,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage=slippage,
                dynamic_slippage=dynamic_slippage,
                position_sizing=position_sizing,
                position_fraction=position_fraction,
                atr_risk_pcts=atr_risk_pcts,
                atr_multipliers=atr_multipliers,
                atr_min_fraction=atr_min_fraction,
                atr_max_fraction=atr_max_fraction,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                optimize_risk=optimize_risk,
            )
            return {"market": market, "symbol": symbol, "status": "success", "result_df": result_df}
        except Exception as e:
            logger.error(f"[{market}/{symbol}] 最適化エラー: {e}", exc_info=True)
            return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}

    # フェーズ1: 最適化計算（並列）
    optimize_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_optimize_task, task): task for task in symbols}
        for future in as_completed(futures):
            optimize_results.append(future.result())

    # フェーズ2: 結果保存（逐次 — JSON書き込みの競合を防止）
    summary = []
    success_count = 0
    error_count = 0
    skip_count = 0
    for res in optimize_results:
        market = res["market"]
        symbol = res["symbol"]
        if res["status"] == "skipped":
            summary.append({"market": market, "symbol": symbol, "status": "skipped"})
            skip_count += 1
            continue
        if res["status"] != "success":
            logger.error(f"[{market}/{symbol}] スキップ（最適化失敗）: {res.get('error')}")
            summary.append(
                {"market": market, "symbol": symbol, "status": "error", "error": res.get("error")}
            )
            error_count += 1
            continue

        result_df = res["result_df"]
        try:
            csv_path = save_optimization_results(result_df, market, symbol)
            json_path = save_optimal_params_json(result_df, market, symbol, sort_by=sort_by)
            logger.info(f"[{market}/{symbol}] 保存完了 CSV={csv_path} JSON={json_path}")
            summary.append({"market": market, "symbol": symbol, "status": "success"})
            success_count += 1
        except Exception as e:
            logger.error(f"[{market}/{symbol}] 保存エラー: {e}", exc_info=True)
            summary.append({"market": market, "symbol": symbol, "status": "error", "error": str(e)})
            error_count += 1

    logger.info(
        f"全銘柄最適化バッチ完了: 成功={success_count} / スキップ={skip_count} / エラー={error_count}"
    )
    return summary


def run_optuna_batch(
    tasks: list,
    model_type: str = "XGBoostModel",
    ensemble: bool = False,
    source: str = "file",
    n_splits: int = 5,
    n_trials: int = 50,
    max_workers: int = 3,
    sort_by: str = "sharpe_ratio",
) -> list[dict[str, Any]]:
    """
    ウォッチリスト全銘柄に Optuna 最適化をバッチ実行し、
    optimal_params.json を更新する（週次スケジューラから呼び出す）。

    tasks は呼び出し元（orchestration 等）で load_target_symbols() して渡す。

    Args:
        tasks: 対象銘柄タスクリスト（SymbolTask または dict）
        n_trials: 銘柄ごとの Optuna 試行回数
        max_workers: 並列数
        その他: ``run_optuna_optimization`` と同様

    Returns:
        各銘柄の結果サマリー list[dict[str, Any]]
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    symbols = tasks
    if not symbols:
        logger.warning("対象銘柄がありません。")
        return []

    logger.info(
        f"Optunaバッチ最適化開始: {len(symbols)}銘柄 / n_trials={n_trials} / 並列={max_workers}"
    )

    def _task(task: Any) -> dict[str, Any]:
        m = getattr(task, "market", None) or task["market"]
        s = getattr(task, "symbol", None) or task["symbol"]
        try:
            result_df = run_optuna_optimization(
                market=m,
                symbol=s,
                model_type=model_type,
                ensemble=ensemble,
                source=source,
                n_splits=n_splits,
                n_trials=n_trials,
                sort_by=sort_by,
            )
            return {"market": m, "symbol": s, "status": "success", "result_df": result_df}
        except Exception as e:
            logger.error(f"[{m}/{s}] Optuna最適化エラー: {e}", exc_info=True)
            return {"market": m, "symbol": s, "status": "error", "error": str(e)}

    # フェーズ1: 並列最適化
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_task, t): t for t in symbols}
        for future in as_completed(futures):
            results.append(future.result())

    # フェーズ2: 逐次保存
    summary, success, error = [], 0, 0
    for res in results:
        m, s = res["market"], res["symbol"]
        if res["status"] != "success":
            summary.append({"market": m, "symbol": s, "status": "error", "error": res.get("error")})
            error += 1
            continue
        try:
            save_optimal_params_json(res["result_df"], m, s, sort_by=sort_by)
            summary.append({"market": m, "symbol": s, "status": "success"})
            success += 1
        except Exception as e:
            logger.error(f"[{m}/{s}] 保存エラー: {e}", exc_info=True)
            summary.append({"market": m, "symbol": s, "status": "error", "error": str(e)})
            error += 1

    logger.info(f"Optunaバッチ最適化完了: 成功={success} / エラー={error}")
    return summary
