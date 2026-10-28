"""
ジョブの締め（recovery_until）の判定

recovery_until: (hour, minute) を持つジョブは、その日のその時刻（JST）から動かさない。
補完の道（#835）と定時の道（#1298）が同じ判定を使う。
"""

from __future__ import annotations

import logging
from datetime import datetime
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
