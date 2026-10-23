"""日次の道のやり直しの「最新の回が今日か」を JST の日付で決めることのテスト（#935）。

本番のコンテナの時計は UTC で、predicted_at は UTC の YYYYMMDD_HHMMSS になる。
UTC の 00:00（JST 09:00）をまたいだやり直しでも、同じ朝の回を今日と見なす。
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.orchestration.jobs import alerting
from src.orchestration.jobs.accuracy_json import production_already_predicted_today
from src.orchestration.jobs.alerting import (
    evaluate_output_invariants_stage,
    fetch_previous_run_stats,
)

_SAME_MORNING = "20261022_224556"  # JST 10-23 の朝の回
_PREVIOUS_MORNING = "20261021_224556"  # JST 10-22 の朝の回


@pytest.fixture(autouse=True)
def _clear_retry_state():
    alerting._previous_day_stats.clear()
    alerting._compare_skipped_days.clear()
    yield


def _utc_clock(*args: int):
    """時計が UTC のコンテナの今を固定する。"""
    now = datetime(*args, tzinfo=timezone.utc)
    return patch("src.orchestration.jobs.common._local_now", return_value=now)


def _latest(predicted_at: str):
    return patch("src.prediction.db.load_latest_prediction_timestamp", return_value=predicted_at)


def test_already_predicted_today_after_jst_0900():
    """JST 10:30 のやり直しでは、同じ朝 (UTC で前の日の日付) の回を今日と見なす。"""
    with _utc_clock(2026, 10, 23, 1, 30), _latest(_SAME_MORNING):
        assert production_already_predicted_today() is True


def test_already_predicted_today_false_on_normal_morning():
    """ふつうの朝 (JST 07:30) は、前の日の朝の回を前の日の回として扱う。"""
    with _utc_clock(2026, 10, 22, 22, 30), _latest(_PREVIOUS_MORNING):
        assert production_already_predicted_today() is False


def test_previous_stats_not_same_morning_after_jst_0900():
    """JST 10:30 のやり直しでは、同じ朝の回を前回とせず None を返し B-0 を出す。"""
    with (
        _utc_clock(2026, 10, 23, 1, 30),
        _latest(_SAME_MORNING),
        patch("src.prediction.db.prediction_results.load_run_stats_at") as stats_at,
    ):
        assert fetch_previous_run_stats() is None
        stats_at.assert_not_called()
        with patch("src.prediction.output_invariants.evaluate_output_invariants") as evaluate:
            evaluate.return_value.violation_ids = []
            evaluate.return_value.as_details.return_value = {"violations": []}
            violation_ids, details = evaluate_output_invariants_stage([], [], [], None)
    assert "B-0" in violation_ids


def test_previous_stats_returns_cached_previous_day_after_jst_0900():
    """同じ JST の日の1回目が退避した前の日の統計があれば、それを返す。"""
    cached = object()
    with _utc_clock(2026, 10, 22, 22, 30), _latest(_PREVIOUS_MORNING):
        with (
            patch(
                "src.prediction.db.prediction_results.load_run_stats_at",
                return_value=([2], [0.01]),
            ),
            patch("src.prediction.output_invariants.build_run_stats", return_value=cached),
        ):
            assert fetch_previous_run_stats() is cached
    with _utc_clock(2026, 10, 23, 1, 30), _latest(_SAME_MORNING):
        assert fetch_previous_run_stats() is cached


def test_previous_stats_previous_day_on_normal_morning():
    """ふつうの朝 (JST 07:30) は、前の日の朝の回の統計を読んで前回とする。"""
    with (
        _utc_clock(2026, 10, 22, 22, 30),
        _latest(_PREVIOUS_MORNING),
        patch(
            "src.prediction.db.prediction_results.load_run_stats_at",
            return_value=([2], [0.01]),
        ) as stats_at,
        patch("src.prediction.output_invariants.build_run_stats", return_value="stats"),
    ):
        assert fetch_previous_run_stats() == "stats"
    stats_at.assert_called_once_with(_PREVIOUS_MORNING, model_version="production")
