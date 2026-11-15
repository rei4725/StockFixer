"""scheduler_cutoff の締めの判定と補完の基準の定時を直に呼んで確かめるテスト（#1833）"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orchestration.scheduler_cutoff import (  # noqa: E402
    is_past_recovery_until,
    scheduled_datetime_for_recovery,
    warn_scheduled_start_past_cutoff,
)

_JST = ZoneInfo("Asia/Tokyo")
_NOTIFY = "src.reporting.discord.discord_utils.send_webhook_text"
_CUTOFF_CONFIG = {"hour": 8, "minute": 50, "recovery_until": (9, 0)}


# --- is_past_recovery_until ---


@pytest.mark.parametrize(
    "now, expected",
    [
        # 締めの1分前（JST）
        (datetime(2026, 3, 2, 8, 59, tzinfo=_JST), False),
        # 締めの直前（59秒）
        (datetime(2026, 3, 2, 8, 59, 59, tzinfo=_JST), False),
        # 締めと等しい時刻で True になる境（>=）
        (datetime(2026, 3, 2, 9, 0, tzinfo=_JST), True),
        # 締めの後
        (datetime(2026, 3, 2, 9, 31, tzinfo=_JST), True),
        # UTC 23:59 は JST の翌日 08:59 で、締めの前
        (datetime(2026, 3, 1, 23, 59, tzinfo=timezone.utc), False),
        # UTC 00:00 は JST 09:00 で、締めと等しい
        (datetime(2026, 3, 2, 0, 0, tzinfo=timezone.utc), True),
        # UTC 08:30 は JST 17:30 で、締めの後（UTC の時と分で比べると締めの前に見える）
        (datetime(2026, 3, 2, 8, 30, tzinfo=timezone.utc), True),
    ],
)
def test_is_past_recovery_until_boundary(now, expected):
    assert is_past_recovery_until(_CUTOFF_CONFIG, now) is expected


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 3, 2, 0, 0, tzinfo=_JST),
        datetime(2026, 3, 2, 23, 59, tzinfo=_JST),
        datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc),
    ],
)
def test_is_past_recovery_until_without_cutoff_is_always_false(now):
    assert is_past_recovery_until({"hour": 8, "minute": 50}, now) is False
    assert is_past_recovery_until({"hour": 8, "minute": 50, "recovery_until": None}, now) is False


# --- scheduled_datetime_for_recovery ---
# 2026-03-02 は月曜（weekday 0）

_WEEKLY_REST = {"hour": 6, "minute": 0, "period": "weekly", "recover_rest_of_week": True}


@pytest.mark.parametrize(
    "now, weekdays, expected",
    [
        # 定時の曜日（水）より前の日（月・火）は補完を考えない
        (datetime(2026, 3, 2, 12, 0, tzinfo=_JST), {2}, None),
        (datetime(2026, 3, 3, 12, 0, tzinfo=_JST), {2}, None),
        # 定時の曜日の当日は、その日の定時
        (datetime(2026, 3, 4, 12, 0, tzinfo=_JST), {2}, datetime(2026, 3, 4, 6, 0, tzinfo=_JST)),
        # 定時の曜日の後の日（土・日）は、その週の定時の曜日の時刻
        (datetime(2026, 3, 7, 12, 0, tzinfo=_JST), {2}, datetime(2026, 3, 4, 6, 0, tzinfo=_JST)),
        (datetime(2026, 3, 8, 23, 0, tzinfo=_JST), {2}, datetime(2026, 3, 4, 6, 0, tzinfo=_JST)),
        # 定時の曜日が複数なら、最も早い曜日を基準にする
        (datetime(2026, 3, 6, 12, 0, tzinfo=_JST), {2, 4}, datetime(2026, 3, 4, 6, 0, tzinfo=_JST)),
        (datetime(2026, 3, 3, 12, 0, tzinfo=_JST), {2, 4}, None),
    ],
)
def test_weekly_rest_of_week(now, weekdays, expected):
    assert scheduled_datetime_for_recovery(_WEEKLY_REST, now, weekdays) == expected


def test_weekly_without_rest_of_week_only_on_scheduled_weekday():
    config = {"hour": 6, "minute": 0, "period": "weekly"}
    # 水曜の定時のジョブは、木曜には補完を考えない
    assert (
        scheduled_datetime_for_recovery(config, datetime(2026, 3, 5, 12, 0, tzinfo=_JST), {2})
        is None
    )
    assert scheduled_datetime_for_recovery(
        config, datetime(2026, 3, 4, 12, 0, tzinfo=_JST), {2}
    ) == datetime(2026, 3, 4, 6, 0, tzinfo=_JST)


@pytest.mark.parametrize(
    "now, weekdays, expected",
    [
        # 定時の曜日（平日）の当日は、その日の定時。秒とマイクロ秒は0にする
        (
            datetime(2026, 3, 2, 9, 31, 45, 123, tzinfo=_JST),
            {0, 1, 2, 3, 4},
            datetime(2026, 3, 2, 8, 50, tzinfo=_JST),
        ),
        # 定時の曜日でない日（土）は None
        (datetime(2026, 3, 7, 9, 0, tzinfo=_JST), {0, 1, 2, 3, 4}, None),
        # 空の weekdays は毎日を表す
        (datetime(2026, 3, 7, 9, 0, tzinfo=_JST), set(), datetime(2026, 3, 7, 8, 50, tzinfo=_JST)),
    ],
)
def test_daily_scheduled_weekdays(now, weekdays, expected):
    config = {"hour": 8, "minute": 50}
    assert scheduled_datetime_for_recovery(config, now, weekdays) == expected


@pytest.mark.parametrize(
    "day, now, expected",
    [
        # day の前の日は None
        (15, datetime(2026, 3, 14, 12, 0, tzinfo=_JST), None),
        # day の当日は、その日の定時
        (15, datetime(2026, 3, 15, 12, 0, tzinfo=_JST), datetime(2026, 3, 15, 7, 0, tzinfo=_JST)),
        # day の後の同じ月は、day の日の定時
        (15, datetime(2026, 3, 31, 12, 0, tzinfo=_JST), datetime(2026, 3, 15, 7, 0, tzinfo=_JST)),
        # day が int でなければ None
        ("15", datetime(2026, 3, 20, 12, 0, tzinfo=_JST), None),
        ("last", datetime(2026, 3, 31, 12, 0, tzinfo=_JST), None),
        (None, datetime(2026, 3, 20, 12, 0, tzinfo=_JST), None),
    ],
)
def test_monthly_day(day, now, expected):
    config = {"hour": 7, "minute": 0, "period": "monthly", "day": day}
    assert scheduled_datetime_for_recovery(config, now, set()) == expected


def test_monthly_without_day_defaults_to_first():
    config = {"hour": 7, "minute": 0, "period": "monthly"}
    assert scheduled_datetime_for_recovery(
        config, datetime(2026, 3, 1, 12, 0, tzinfo=_JST), set()
    ) == datetime(2026, 3, 1, 7, 0, tzinfo=_JST)


def test_monthly_respects_weekdays():
    config = {"hour": 7, "minute": 0, "period": "monthly", "day": 1}
    # 2026-03-07 は土曜。平日だけのジョブは day の後でも補完を考えない
    assert (
        scheduled_datetime_for_recovery(
            config, datetime(2026, 3, 7, 12, 0, tzinfo=_JST), {0, 1, 2, 3, 4}
        )
        is None
    )


# --- warn_scheduled_start_past_cutoff ---


def test_warn_logs_and_notifies_in_jst(caplog):
    # UTC 00:31 は JST 09:31
    now = datetime(2026, 3, 2, 0, 31, tzinfo=timezone.utc)
    with patch(_NOTIFY) as mock_notify, caplog.at_level("WARNING", logger="scheduler"):
        warn_scheduled_start_past_cutoff("daily_auto_order", _CUTOFF_CONFIG, "2026-03-02", now)

    expected = (
        "定時の回を飛ばしました: daily_auto_order（period=2026-03-02）"
        " 走り出しが 09:31 JST で、締め 09:00 を過ぎています"
    )
    assert expected in caplog.text
    mock_notify.assert_called_once_with(f"⚠️ {expected}")


def test_warn_does_not_raise_when_notify_fails(caplog):
    now = datetime(2026, 3, 2, 9, 31, tzinfo=_JST)
    with (
        patch(_NOTIFY, side_effect=RuntimeError("webhook down")) as mock_notify,
        caplog.at_level("WARNING", logger="scheduler"),
    ):
        warn_scheduled_start_past_cutoff("daily_auto_order", _CUTOFF_CONFIG, "2026-03-02", now)

    mock_notify.assert_called_once()
    assert "定時の回を飛ばしました" in caplog.text
    assert "締めを過ぎた定時の回の知らせに失敗: daily_auto_order" in caplog.text
