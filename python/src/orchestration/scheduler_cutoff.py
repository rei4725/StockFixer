"""
ジョブの締め（recovery_until）の判定と、補完の基準の定時の割り出し

recovery_until: (hour, minute) を持つジョブは、その日のその時刻（JST）から動かさない。
補完の道（#835）と定時の道（#1298）が同じ判定を使う。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger("scheduler")

_JST = ZoneInfo("Asia/Tokyo")


def is_past_recovery_until(config: dict[str, Any], now: datetime) -> bool:
    """ジョブの締めを過ぎていれば True。締めの欄が無いジョブは常に False。"""
    recovery_until = config.get("recovery_until")
    if recovery_until is None:
        return False
    until_hour, until_minute = recovery_until
    # コンテナの時刻は UTC のため、JST に直してから比べる
    now_jst = now.astimezone(_JST)
    cutoff = now_jst.replace(hour=until_hour, minute=until_minute, second=0, microsecond=0)
    return now_jst >= cutoff


def scheduled_datetime_for_recovery(
    config: dict[str, Any], now: datetime, weekdays: set[int]
) -> datetime | None:
    """今の補完の判定が基準にする定時を返す。今が補完を考えない日なら None。

    weekdays は設定の day_of_week の曜日の番号で、空は毎日を表す。

    - recover_rest_of_week の週次ジョブは、ISO 週の最初の定時の曜日から日曜まで補完を考え、
      基準はその週の定時の曜日の時刻とする（#1534）
    - monthly は設定の day の日とそれ以後の同じ月に補完を考え、基準はその day の日の時刻とする
    - ほかのジョブは定時の曜日だけ補完を考え、基準はその日の時刻とする
    """
    scheduled = now
    if weekdays and config.get("period") == "weekly" and config.get("recover_rest_of_week", False):
        days_after_scheduled = now.weekday() - min(weekdays)
        if days_after_scheduled < 0:
            return None
        scheduled = now - timedelta(days=days_after_scheduled)
    elif weekdays and now.weekday() not in weekdays:
        return None

    if config.get("period", "daily") == "monthly":
        scheduled_day = config.get("day", 1)
        if not isinstance(scheduled_day, int) or now.day < scheduled_day:
            return None
        scheduled = now.replace(day=scheduled_day)

    return scheduled.replace(hour=config["hour"], minute=config["minute"], second=0, microsecond=0)


def warn_scheduled_start_past_cutoff(
    job_id: str, config: dict[str, Any], period_key: str, now: datetime
) -> None:
    """定時の回が締めを過ぎて走り出したことを WARNING と Discord で知らせる（#1298）。"""
    until_hour, until_minute = config["recovery_until"]
    started = now.astimezone(_JST).strftime("%H:%M")
    message = (
        f"定時の回を飛ばしました: {job_id}（period={period_key}）"
        f" 走り出しが {started} JST で、締め {until_hour:02d}:{until_minute:02d} を過ぎています"
    )
    logger.warning(message)
    try:
        from src.reporting.discord.discord_utils import send_webhook_text

        send_webhook_text(f"⚠️ {message}")
    except Exception:
        logger.error(f"締めを過ぎた定時の回の知らせに失敗: {job_id}", exc_info=True)
