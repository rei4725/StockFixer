"""Discord API 層向けの問い合わせサービス。"""

from __future__ import annotations

import csv
import json
import os

import pandas as pd

from src.domain.types import PredictionResult, ShapFeatureContribution, SignalSnapshot
from src.orchestration.types import SchedulerJobStatus
from src.reporting.types import (
    MarketPredictionSnapshot,
    MonthlyReportSummary,
    WatchlistPredictionRow,
    WatchlistPredictionView,
)
from src.utils.data_path_utils import get_monitor_list_path, get_results_dir
from src.utils.db import (
    load_latest_prediction_timestamp,
    load_prediction_markets,
    load_prediction_results,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _to_prediction_results(df: pd.DataFrame | None) -> list[PredictionResult]:
    if df is None or df.empty:
        return []
    return [PredictionResult.from_dataframe_row(row) for _, row in df.iterrows()]


def get_ranked_prediction_results(
    market: str,
    rank_type: str,
    predicted_at: str | None = None,
) -> list[PredictionResult]:
    if rank_type == "top10":
        df = load_prediction_results(predicted_at=predicted_at, market=market, top_n=10)
    elif rank_type == "worst10":
        df = load_prediction_results(predicted_at=predicted_at, market=market, worst_n=10)
    else:
        df = load_prediction_results(predicted_at=predicted_at, market=market)
    return _to_prediction_results(df)


def get_latest_market_prediction_snapshots() -> tuple[str | None, list[MarketPredictionSnapshot]]:
    latest_ts = load_latest_prediction_timestamp()
    if not latest_ts:
        return None, []

    snapshots: list[MarketPredictionSnapshot] = []
    for market in sorted(load_prediction_markets(latest_ts) or []):
        snapshots.append(
            MarketPredictionSnapshot(
                market=market,
                top_results=get_ranked_prediction_results(market, "top10", latest_ts),
                worst_results=get_ranked_prediction_results(market, "worst10", latest_ts),
            )
        )
    return latest_ts, snapshots


def get_watchlist_prediction_view() -> WatchlistPredictionView:
    from src.prediction.predict_single import predict_single_stock

    watchlist_path = get_monitor_list_path()
    rows: list[WatchlistPredictionRow] = []
    try:
        with open(watchlist_path, encoding="utf-8") as file_handle:
            reader = csv.reader(file_handle)
            for row in reader:
                if len(row) < 2:
                    continue
                market, symbol = row[0], row[1]
                result_df = predict_single_stock(market, symbol)
                if result_df is None or result_df.empty:
                    rows.append(
                        WatchlistPredictionRow(
                            symbol=symbol,
                            current_price=None,
                            avg_pred_price=None,
                            diff_ratio=None,
                        )
                    )
                    continue
                record = result_df.iloc[0]
                try:
                    diff_ratio = (
                        float(record["avg_pred_price"]) - float(record["current_price"])
                    ) / float(record["current_price"])
                except (ValueError, TypeError, ZeroDivisionError, KeyError):
                    diff_ratio = None
                rows.append(
                    WatchlistPredictionRow(
                        symbol=str(record["symbol"]),
                        current_price=float(record["current_price"]),
                        avg_pred_price=float(record["avg_pred_price"]),
                        diff_ratio=diff_ratio,
                    )
                )
    except Exception as exc:
        logger.error("監視対象予測処理で例外発生", exc_info=True)
        return WatchlistPredictionView(error_message=f"[エラー] 監視対象予測処理で例外: {exc}")

    return WatchlistPredictionView(rows=rows)


def get_signal_snapshot(market: str, symbol: str, explain: bool = False) -> SignalSnapshot | None:
    from src.prediction.predict_single import explain_prediction_shap, predict_single_stock

    result_df = predict_single_stock(market, symbol)
    if result_df is None or result_df.empty:
        return None

    prediction = PredictionResult.from_dataframe_row(result_df.iloc[0])
    if not explain:
        return SignalSnapshot(prediction=prediction)

    shap_result = explain_prediction_shap(market, symbol, top_n=5)
    if not shap_result:
        return SignalSnapshot(prediction=prediction)

    features = [
        ShapFeatureContribution(feature=item["feature"], shap_value=float(item["shap_value"]))
        for item in shap_result.get("top_features", [])
    ]
    return SignalSnapshot(
        prediction=prediction,
        shap_direction=shap_result.get("direction"),
        top_features=features,
    )


def get_scheduler_job_statuses(state_file_path: str | None = None) -> list[SchedulerJobStatus]:
    state_path = state_file_path or os.path.join(get_results_dir(), "scheduler_queue_state.json")
    with open(state_path, encoding="utf-8") as file_handle:
        state = json.load(file_handle)

    events = state.get("events", [])
    last_runs: dict[str, str] = {}
    last_status: dict[str, str] = {}
    for event in reversed(events):
        job_id = event.get("job_id", "")
        if job_id and job_id not in last_runs:
            last_runs[job_id] = event.get("finished_at") or event.get("started_at")
            last_status[job_id] = event.get("status", "不明")

    job_labels = {
        "daily_pipeline": "日次 (daily_pipeline)",
        "weekly_model_training": "週次 (weekly_model_training)",
    }
    return [
        SchedulerJobStatus(
            job_id=job_id,
            label=label,
            last_run_at=last_runs.get(job_id),
            status=last_status.get(job_id, "-"),
        )
        for job_id, label in job_labels.items()
    ]


def get_monthly_report_summary(target_month: str | None = None) -> MonthlyReportSummary:
    """月次KPIサマリーを取得する。Discord /monthlyreport コマンド向け。"""
    from src.reporting.monthly import run_monthly_report

    return run_monthly_report(target_month=target_month)
