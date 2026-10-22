"""日次の道を頭からやり直したときに、採点の JSON を0件で上書きしないことのテスト（#904）。

prediction_results は Delete-Insert のスナップショットテーブルなので、[3/6] の保存の後に
[5/6] で落ちて頭からやり直すと、[2/6] は今日の予想（まだ採点の無い行）を読む。
そのときは採点と JSON の書き出しを飛ばし、1回目の JSON を残す。
"""

import json
from contextlib import ExitStack
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_SYMBOLS = ["7203", "6758"]


def _stamp(day: datetime) -> str:
    return day.strftime("%Y%m%d") + "_073000"


class _FakePredictionTable:
    """prediction_results（Delete-Insert）と prediction_accuracy の代わり。"""

    def __init__(self, production_at: str, scored_at: str):
        self.production_at = production_at
        self.scored_at = scored_at

    def latest_timestamp(self, model_version=None):
        return self.production_at

    def load_results(self, limit=None, model_version=None, **kwargs):
        if model_version != "production":
            return pd.DataFrame()
        return pd.DataFrame(
            {"market": "jp", "symbol": _SYMBOLS, "predicted_at": self.production_at}
        )

    def load_accuracy(self, horizon=1, limit=500, **kwargs):
        return pd.DataFrame(
            {
                "market": "jp",
                "symbol": _SYMBOLS,
                "model_name": "production",
                "predicted_at": self.scored_at,
                "horizon": horizon,
                "predicted_ratio": [0.01, -0.02],
                "actual_ratio": [0.02, 0.01],
                "direction_match": [True, False],
            }
        )

    def save_today(self, *args, **kwargs):
        """[3/6] の保存: production の回を今日の予想に置き換える。"""
        self.production_at = _stamp(datetime.now())


@pytest.fixture
def results_dir(tmp_path):
    with patch("src.prediction.accuracy_report.get_results_dir", return_value=str(tmp_path)):
        yield tmp_path


def _run_pipeline(table: _FakePredictionTable, notify_side_effect=None):
    """外への呼び出しを止めて日次の道を1回呼び、run_accuracy_check の mock を返す。"""
    from src.orchestration.scheduler import run_daily_pipeline

    with ExitStack() as stack:

        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        p("src.watchlist.batch_runner.load_target_symbols", return_value=[])
        p("src.market_data.pipeline.run_batch_pipeline", return_value=MagicMock())
        p("src.prediction.db.load_latest_prediction_timestamp", new=table.latest_timestamp)
        p("src.prediction.db.load_prediction_results", new=table.load_results)
        p("src.prediction.db.load_prediction_accuracy", new=table.load_accuracy)
        accuracy_check = p("src.prediction.prediction_pipeline.run_accuracy_check")
        p("src.reporting.discord.discord_utils.send_accuracy_summary")
        p("src.reporting.discord.discord_utils.send_daily_pipeline_error")
        p("src.prediction.predict_unified.preload_models", return_value=[])
        p("src.orchestration.jobs.daily.fetch_previous_run_stats", return_value=None)
        p("src.prediction.prediction_pipeline.predict_all_unified", return_value=[])
        p(
            "src.prediction.prediction_pipeline.output_top_worst_results",
            side_effect=table.save_today,
        )
        p("src.orchestration.jobs.daily.evaluate_output_invariants_stage", return_value=([], {}))
        p("src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[])
        p("src.orchestration.jobs.daily.run_daily_drift_check")
        p(
            "src.reporting.discord.discord_utils.send_daily_pipeline_completion",
            side_effect=notify_side_effect,
        )
        p("src.orchestration.jobs.daily.evaluate_and_notify_alerts")
        p("src.orchestration.jobs.daily.append_accuracy_json_sectors")
        try:
            run_daily_pipeline()
        finally:
            calls = accuracy_check.call_count
    return calls


def _read_json(results_dir) -> dict:
    files = list((results_dir / "accuracy").glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_normal_morning_scores_and_writes_json(results_dir):
    """前の日の予想が残っているふつうの朝は、今までどおり採点して JSON を書く。"""
    yesterday = _stamp(datetime.now() - timedelta(days=1))
    table = _FakePredictionTable(production_at=yesterday, scored_at=yesterday)

    calls = _run_pipeline(table)

    assert calls == 2  # production と challenger
    report = _read_json(results_dir)
    assert report["models"]["production"]["1"]["n_scored"] == 2


def test_skips_scoring_when_production_already_today(results_dir):
    """production の回が既に今日なら、run_accuracy_check を呼ばず JSON も書き換えない。"""
    today = _stamp(datetime.now())
    yesterday = _stamp(datetime.now() - timedelta(days=1))
    table = _FakePredictionTable(production_at=today, scored_at=yesterday)
    accuracy_dir = results_dir / "accuracy"
    accuracy_dir.mkdir()
    previous = accuracy_dir / f"{datetime.now():%Y-%m-%d}.json"
    previous.write_text('{"models": {"production": {"1": {"n_scored": 5}}}}', encoding="utf-8")

    calls = _run_pipeline(table)

    assert calls == 0
    assert json.loads(previous.read_text(encoding="utf-8")) == {
        "models": {"production": {"1": {"n_scored": 5}}}
    }


def test_retry_after_notify_failure_keeps_first_n_scored(results_dir):
    """[5/6] で落ちて頭からやり直しても、1回目の JSON の n_scored が残る。"""
    yesterday = _stamp(datetime.now() - timedelta(days=1))
    table = _FakePredictionTable(production_at=yesterday, scored_at=yesterday)

    with pytest.raises(RuntimeError):
        _run_pipeline(table, notify_side_effect=RuntimeError("Discord 400"))
    assert _read_json(results_dir)["models"]["production"]["1"]["n_scored"] == 2

    # scheduler_queue の同じ回の中のやり直しと同じく、頭から呼び直す
    calls = _run_pipeline(table)

    assert calls == 0
    assert _read_json(results_dir)["models"]["production"]["1"]["n_scored"] == 2
