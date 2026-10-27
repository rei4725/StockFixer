"""日次の道のやり直しで、今日の回の行だけを採点から外し前の日の回の行は採点することのテスト（#1255）。

1回目の [1/6] で us の足が取れず us の前の日の予想が採点されずに残ったまま、[3/6] で
jp だけが今日の回に入れ替わり、再起動で頭からやり直すと、最新の回は今日になる。
そのとき [2/6] を丸ごと飛ばすと、us の前の日の行は [3/6] の上書きで採点されずに消える。
"""

from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

_DAILY = "src.orchestration.jobs.daily"
_DISCORD = "src.reporting.discord.discord_utils"

_TODAY_RUN = "20261026_233359"  # JST 10-27 08:33 の回（1回目）
_PREVIOUS_RUN = "20261025_224351"  # JST 10-26 07:43 の回（前の日）


def _row(market: str, symbol: str, predicted_at: str) -> dict:
    return {
        "market": market,
        "symbol": symbol,
        "predicted_at": predicted_at,
        "avg_pred_price": 101.0,
        "current_price": 100.0,
    }


def _ohlcv(market: str, symbol: str) -> pd.DataFrame:
    """次の足が DB に在る形。jp は 10-27 の足まで在り、外さなければ今日の行も採点できる。"""
    days = ["2026-10-23", "2026-10-26", "2026-10-27"]
    return pd.DataFrame({"Close": [100.0, 102.0, 103.0]}, index=pd.to_datetime(days))


def _run_pipeline(snapshot: dict[str, list[dict]]):
    """JST 10-27 09:04 に日次の道を1回呼び、外への呼び出しの mock を返す。"""
    from src.orchestration.jobs.daily import run_daily_pipeline

    saved: list[dict] = []

    def load_results(limit=5000, model_version=None, raise_on_error=False):
        return pd.DataFrame(snapshot.get(model_version, []))

    def latest_timestamp(model_version=None):
        rows = snapshot.get(model_version, [])
        return max((r["predicted_at"] for r in rows), default=None)

    with ExitStack() as stack:

        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        now = datetime(2026, 10, 27, 0, 4, tzinfo=timezone.utc)
        p("src.orchestration.jobs.common._local_now", return_value=now)
        p("src.prediction.db.load_latest_prediction_timestamp", new=latest_timestamp)
        p("src.prediction.db.load_prediction_results", new=load_results)
        p("src.prediction.db.save_prediction_accuracy", new=saved.extend)
        p("src.prediction.db.load_drift_summary", return_value=pd.DataFrame())
        p("src.utils.db.load_raw_ohlcv", new=_ohlcv)
        mocks = {
            "summary": p(f"{_DISCORD}.send_accuracy_summary"),
            "prepare": p("src.prediction.accuracy_report.prepare_daily_accuracy_report"),
            "append": p(f"{_DAILY}.append_accuracy_json_sectors"),
            "webhook": p(f"{_DISCORD}.send_webhook_text"),
            "predict": p("src.prediction.prediction_pipeline.predict_all_unified", return_value=[]),
        }
        p(f"{_DISCORD}.send_daily_pipeline_completion")
        p(f"{_DISCORD}.send_daily_pipeline_error")
        p("src.prediction.prediction_pipeline.output_top_worst_results")
        p("src.prediction.predict_unified.preload_models", return_value=[])
        p("src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[])
        p("src.watchlist.batch_runner.load_target_symbols", return_value=[])
        p("src.market_data.pipeline.run_batch_pipeline")
        p("src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter")
        p(f"{_DAILY}._raise_if_fetch_mostly_unavailable")
        p(f"{_DAILY}.fetch_previous_run_stats", return_value=None)
        p(f"{_DAILY}.evaluate_output_invariants_stage", return_value=([], {}))
        p(f"{_DAILY}.run_daily_drift_check")
        p(f"{_DAILY}.record_health_degraded_streak")
        p(f"{_DAILY}.evaluate_and_notify_alerts")
        p(
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository"
        )
        run_daily_pipeline()
    mocks["saved"] = saved
    return mocks


def _scored(mocks) -> set[tuple[str, str, str, str]]:
    return {(r["model_name"], r["market"], r["symbol"], r["predicted_at"]) for r in mocks["saved"]}


def _mixed() -> list[dict]:
    """jp は今日の回に入れ替わり、us は前の日の回の行が残っている形。"""
    return [_row("jp", "7203", _TODAY_RUN), _row("us", "A", _PREVIOUS_RUN)]


def test_rerun_scores_previous_day_rows_and_skips_today_rows():
    """やり直しでも us の前の日の行を採点し、jp の今日の行は採点しない（challenger も同じ）。"""
    mocks = _run_pipeline({"production": _mixed(), "challenger": _mixed()})

    assert _scored(mocks) == {
        ("production", "us", "A", _PREVIOUS_RUN),
        ("challenger", "us", "A", _PREVIOUS_RUN),
    }
    mocks["webhook"].assert_not_called()
    mocks["predict"].assert_called_once()


def test_rerun_keeps_first_accuracy_json():
    """やり直しでは採点の JSON を書き直さず、1回目の JSON を残す（0件や今日の行で上書きしない）。"""
    mocks = _run_pipeline({"production": _mixed()})

    mocks["prepare"].assert_not_called()
    mocks["append"].assert_called_once_with(None)
    mocks["summary"].assert_not_called()


def test_normal_morning_scores_all_previous_day_rows_and_writes_json():
    """前の日の回だけがあるふつうの朝は、今までどおり全部を採点し JSON を書く。"""
    rows = [_row("jp", "7203", _PREVIOUS_RUN), _row("us", "A", _PREVIOUS_RUN)]
    mocks = _run_pipeline({"production": rows})

    assert _scored(mocks) == {
        ("production", "jp", "7203", _PREVIOUS_RUN),
        ("production", "us", "A", _PREVIOUS_RUN),
    }
    mocks["summary"].assert_called_once()
    mocks["prepare"].assert_called_once()
    mocks["prepare"].return_value.write.assert_called_once_with(sector_getter=None)
    mocks["append"].assert_called_once_with(mocks["prepare"].return_value)


def test_rerun_with_all_rows_today_scores_nothing_and_keeps_json():
    """全銘柄が今日の回に入れ替わったやり直し（#904）は、何も採点せず JSON を残す。"""
    rows = [_row("jp", "7203", _TODAY_RUN), _row("us", "A", _TODAY_RUN)]
    mocks = _run_pipeline({"production": rows, "challenger": rows})

    assert mocks["saved"] == []
    mocks["prepare"].assert_not_called()
    mocks["append"].assert_called_once_with(None)
