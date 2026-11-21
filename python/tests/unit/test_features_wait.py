"""特徴量の待ちの原因の分け方のテスト（#2086）"""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.orchestration.jobs.features_wait import (
    DAILY_RUN_WITHOUT_MARKER,
    WAIT_NEXT_DAILY_RUN,
    classify_features_wait,
    next_daily_run,
)

_JST = ZoneInfo("Asia/Tokyo")


def _jst(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=_JST)


def test_next_daily_run_skips_weekend_and_holiday():
    # 金曜の夜の次は、土日と勤労感謝の日（11-23）を越えた火曜の朝
    assert next_daily_run(_jst(20, 22)) == _jst(24, 7, 30)


def test_next_daily_run_same_morning_before_run_time():
    assert next_daily_run(_jst(24, 6)) == _jst(24, 7, 30)


def test_marker_written_this_morning_waits_for_next_run():
    wait = classify_features_wait(_jst(20, 7, 39), _jst(20, 22))
    assert wait.cause == WAIT_NEXT_DAILY_RUN
    assert wait.next_daily_run == _jst(24, 7, 30)


def test_daily_run_after_marker_is_failure():
    wait = classify_features_wait(_jst(20, 7, 39), _jst(24, 22))
    assert wait.cause == DAILY_RUN_WITHOUT_MARKER
    assert wait.next_daily_run == _jst(25, 7, 30)


def test_missing_written_at_is_failure():
    assert classify_features_wait(None, _jst(21, 22)).cause == DAILY_RUN_WITHOUT_MARKER


def test_unreadable_marker_time_is_none(tmp_path, monkeypatch):
    from src.prediction.backtest_promotion import state as st

    monkeypatch.setenv("STOCKFIXER_RESULTS_DIR", str(tmp_path))
    assert st.load_features_written_at() is None
    path = st._features_version_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"version": "abc", "written_at": "not a time"}', encoding="utf-8")
    assert st.load_features_written_at() is None
    path.write_text('{"version": "abc"}', encoding="utf-8")
    assert st.load_features_written_at() is None


def test_utc_times_are_read_in_jst():
    # 金曜 07:39 JST は木曜 22:39 UTC。UTC の日付で読まない
    written = datetime(2026, 11, 19, 22, 39, tzinfo=ZoneInfo("UTC"))
    now = datetime(2026, 11, 22, 13, 0, tzinfo=ZoneInfo("UTC"))
    assert classify_features_wait(written, now).cause == WAIT_NEXT_DAILY_RUN
