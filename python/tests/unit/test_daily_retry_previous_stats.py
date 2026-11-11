"""日次の道を頭からやり直したときに、今日の回を前回として急変の見張りで比べないことのテスト（#920）。

prediction_results は Delete-Insert のスナップショットテーブルなので、[3/6] の保存の後に
[5/6] で落ちて頭からやり直すと、最新の回は同じ日の1回目の予想になる。それを前回として
B-1〜B-3 を比べると、ほぼ同じ予想どうしの比べになって鳴らない。
"""

import logging
from contextlib import ExitStack
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.jobs import alerting
from src.prediction.types import PredictionResult

_LOADED = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]


def _stamp(day: datetime) -> str:
    # 今の時刻のまま書く（今日の判定は JST の日付なので、決まった時刻だと UTC の夕方に外れる）
    return day.strftime("%Y%m%d_%H%M%S")


def _rows(count: int):
    return [
        PredictionResult(
            market="jp",
            symbol=str(7000 + i),
            current_price=1000.0 + i,
            avg_pred_price=1010.0 + i,
            diff_ratio=0.001 * (i - count / 2),
            model_count=2,
        )
        for i in range(count)
    ]


class _FakePredictionTable:
    """prediction_results（Delete-Insert）の production の回の代わり。"""

    def __init__(self, predicted_at: str, rows: list, clock=datetime.now):
        self.predicted_at = predicted_at
        self.rows = rows
        self.clock = clock
        self.stats_reads: list[str] = []

    def latest_timestamp(self, model_version=None):
        return self.predicted_at

    def run_stats_at(self, predicted_at, model_version="production"):
        self.stats_reads.append(predicted_at)
        if predicted_at != self.predicted_at:
            return None
        return [r.model_count for r in self.rows], [r.diff_ratio for r in self.rows]

    def save_today(self, rows, *args, **kwargs):
        """[3/6] の保存: production の回を今日の予想に置き換える。"""
        self.predicted_at = _stamp(self.clock())
        self.rows = list(rows)


def _assert_written_between(table: _FakePredictionTable, before: str, after: str) -> None:
    """道の前と後の時刻の間に書かれたこと（今日の回で上書きされたこと）を確かめる。

    秒まで今の時刻と等しいかで比べると、道が秒の境目をまたいだときに1秒ずれて落ちる（#1721）。
    %Y%m%d_%H%M%S は桁が揃っているので、文字列の大小が時刻の前後と一致する。
    """
    assert before <= table.predicted_at <= after


class _TickingClock:
    """呼ばれるたびに1秒進む時計。道の中と後で秒がずれるときの代わり。"""

    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        current = self.now
        self.now = current + timedelta(seconds=1)
        return current


@pytest.fixture(autouse=True)
def _clear_retry_state():
    for name in ("_previous_day_stats", "_compare_skipped_days"):
        state = getattr(alerting, name, None)
        if state is not None:
            state.clear()
    yield


def _run_pipeline(table: _FakePredictionTable, output_rows, notify_side_effect=None):
    """外への呼び出しを止めて日次の道を1回呼び、運用アラートへ渡した引数を返す。"""
    from src.orchestration.jobs.daily import run_daily_pipeline

    with ExitStack() as stack:

        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        p("src.watchlist.batch_runner.load_target_symbols", return_value=[])
        p("src.market_data.pipeline.run_batch_pipeline", return_value=MagicMock())
        p("src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter")
        p("src.orchestration.jobs.daily.production_already_predicted_today", return_value=True)
        p("src.prediction.prediction_pipeline.run_accuracy_check")
        p("src.prediction.db.load_latest_prediction_timestamp", new=table.latest_timestamp)
        p("src.prediction.db.prediction_results.load_run_stats_at", new=table.run_stats_at)
        p("src.reporting.discord.discord_utils.send_daily_pipeline_error")
        p("src.prediction.predict_unified.preload_models", return_value=_LOADED)
        p("src.prediction.prediction_pipeline.predict_all_unified", return_value=output_rows)
        p(
            "src.prediction.prediction_pipeline.output_top_worst_results",
            side_effect=lambda rows, **kwargs: table.save_today(rows),
        )
        p("src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[])
        p("src.orchestration.jobs.daily.run_daily_drift_check")
        p(
            "src.reporting.discord.discord_utils.send_daily_pipeline_completion",
            side_effect=notify_side_effect,
        )
        alerts = p("src.orchestration.jobs.daily.evaluate_and_notify_alerts")
        p("src.orchestration.jobs.daily.append_accuracy_json_sectors")
        run_daily_pipeline()
    violation_ids, details = alerts.call_args.args
    return violation_ids, details


def test_normal_morning_compares_with_previous_day():
    """前の日の予想が残っているふつうの朝は、前の日の回と比べて B-1 を出す。"""
    table = _FakePredictionTable(_stamp(datetime.now() - timedelta(days=1)), _rows(705))

    violation_ids, details = _run_pipeline(table, _rows(100))

    assert details["compared_with_previous"] is True
    assert violation_ids.count("B-1") == 1
    assert "B-0" not in violation_ids


def test_does_not_compare_with_today_run(caplog):
    """最新の回が既に今日なら、それを前回として B-1〜B-3 を比べず、飛ばしたことを出す。"""
    table = _FakePredictionTable(_stamp(datetime.now()), _rows(100))

    with caplog.at_level(logging.WARNING):
        violation_ids, details = _run_pipeline(table, _rows(100))

    assert table.stats_reads == []
    assert details["compared_with_previous"] is False
    assert "B-0" in violation_ids
    assert any(v["id"] == "B-0" for v in details["violations"])
    assert any(
        r.levelno == logging.WARNING and "急変の見張り" in r.getMessage() for r in caplog.records
    )


def _retry_after_notify_failure_still_fires_b1(clock) -> None:
    table = _FakePredictionTable(_stamp(clock() - timedelta(days=1)), _rows(705), clock=clock)

    before = _stamp(clock())
    with pytest.raises(RuntimeError):
        _run_pipeline(table, _rows(100), notify_side_effect=RuntimeError("Discord 400"))
    after = _stamp(clock())
    _assert_written_between(table, before, after)

    # scheduler_queue の同じ回の中のやり直しと同じく、頭から呼び直す
    violation_ids, details = _run_pipeline(table, _rows(100))

    assert details["compared_with_previous"] is True
    assert violation_ids.count("B-1") == 1
    b1 = next(v for v in details["violations"] if v["id"] == "B-1")
    assert "705" in b1["description"]


def test_retry_after_notify_failure_still_fires_b1():
    """[5/6] で落ちて頭から呼び直しても、2回目も前の日の回と比べて B-1 を出す。"""
    _retry_after_notify_failure_still_fires_b1(datetime.now)


def test_retry_after_notify_failure_passes_across_second_boundary():
    """道の中で書いた時刻と道の後の時刻が秒の境目でずれても、上書きの確かめは通る（#1721）。"""
    start = datetime.now().replace(microsecond=999_999)
    _retry_after_notify_failure_still_fires_b1(_TickingClock(start))


def test_written_between_fails_when_previous_day_remains():
    """1回目の後に predicted_at が前の日のままなら、上書きの確かめは落ちる。"""
    now = datetime.now()
    table = _FakePredictionTable(_stamp(now - timedelta(days=1)), _rows(705))

    with pytest.raises(AssertionError):
        _assert_written_between(table, _stamp(now), _stamp(now + timedelta(seconds=1)))
