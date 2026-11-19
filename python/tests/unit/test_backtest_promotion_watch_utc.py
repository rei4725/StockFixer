"""
夜間の仮の入れ替えの見張りの、predicted_at の時計の読み方のテスト（#1963）

predicted_at はプロセスの時計（本番は UTC）の YYYYMMDD_HHMMSS である。
テストを流す機械の時差によらないよう、_local_now を UTC と JST の時計に差し替えて確かめる。
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.orchestration.jobs import backtest_promotion_watch as job

_UTC_NOW = datetime(2026, 10, 2, 0, 0, tzinfo=timezone.utc)
_JST_NOW = datetime(2026, 10, 2, 9, 0, tzinfo=timezone(timedelta(hours=9)))


def _utc_clock():
    return patch.object(job, "_local_now", return_value=_UTC_NOW)


def _jst_clock():
    return patch.object(job, "_local_now", return_value=_JST_NOW)


class TestPredictedAtJst:
    def test_utc_clock_moves_to_next_jst_day(self):
        with _utc_clock():
            got = job._predicted_at_jst("20261001_223000")
        assert got is not None
        assert got.utcoffset() == timedelta(hours=9)
        assert (got.year, got.month, got.day, got.hour, got.minute) == (2026, 10, 2, 7, 30)

    def test_jst_clock_keeps_same_time(self):
        with _jst_clock():
            got = job._predicted_at_jst("20261001_223000")
        assert got is not None
        assert got.utcoffset() == timedelta(hours=9)
        assert (got.year, got.month, got.day, got.hour, got.minute) == (2026, 10, 1, 22, 30)

    @pytest.mark.parametrize("value", ["abc", "", None])
    def test_unreadable_value_is_none(self, value):
        with _utc_clock():
            assert job._predicted_at_jst(value) is None


class TestScoredRowsSinceSwapUtcClock:
    _WATCH = {"swapped_at": "2026-10-01T22:00:00+09:00", "swap_date": "2026-10-01"}

    def _rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # 07:30 JST の朝の回（UTC では入れ替えと同じ日付の 22:30）→ 残る
                {
                    "market": "jp",
                    "symbol": "7203",
                    "predicted_at": "20261001_223000",
                    "actual_ratio": 0.01,
                },
                # 21:00 JST（入れ替えの前）→ 落ちる
                {
                    "market": "jp",
                    "symbol": "6758",
                    "predicted_at": "20261001_120000",
                    "actual_ratio": 0.02,
                },
                # 読めない predicted_at → 落ちる
                {"market": "jp", "symbol": "9984", "predicted_at": "abc", "actual_ratio": 0.03},
                # actual_ratio が欠けている → 落ちる
                {
                    "market": "jp",
                    "symbol": "8306",
                    "predicted_at": "20261001_223000",
                    "actual_ratio": None,
                },
            ]
        )

    def test_keeps_next_morning_run_and_drops_the_rest(self):
        with _utc_clock(), patch.object(job, "load_prediction_accuracy", return_value=self._rows()):
            got = job._scored_rows_since_swap(self._WATCH)
        assert got["symbol"].tolist() == ["7203"]
        assert got["_day"].tolist() == ["2026-10-02"]

    def test_counted_as_next_jst_day(self):
        with _utc_clock(), patch.object(job, "load_prediction_accuracy", return_value=self._rows()):
            rows = job._scored_rows_since_swap(self._WATCH)
        counted, skipped = job._split_watch_days(rows, need_symbols=1)
        assert counted == ["2026-10-02"]
        assert skipped == []
