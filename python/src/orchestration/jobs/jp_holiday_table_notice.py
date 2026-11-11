"""東証の休場日の表が切れる前の知らせ（#1663）。

表に無い年は土日だけを除く形に黙って戻り、休場日の発注の止めと足の遅れの線が
効かなくなる。表の最後の年の 10/1 から、次の年の表を足すまで毎営業日に1回知らせる。
8:50 の自動発注と 16:00 のルールシグナルの両方が通るので、送れた日を results/ の印に
残して、やり直しや起き直しでも1日2通にしない。送れなかった日は印を残さず、次の回が送る。
知らせの失敗は発注の判定に響かせない。
"""

import os
import threading
from datetime import date

from src.domain.jp_market_holidays import missing_jp_holiday_table_year
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MARKER_NAME = "jp_holiday_table_notice_day.txt"
_notice_lock = threading.Lock()


def _marker_path() -> str:
    return os.path.join(get_results_dir(), _MARKER_NAME)


def _read_notified_day() -> str:
    try:
        with open(_marker_path(), encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _mark_notified(today: date) -> None:
    ensure_dir(get_results_dir())
    with open(_marker_path(), "w", encoding="utf-8") as f:
        f.write(today.isoformat())


def _send_notice(year: int) -> bool:
    from src.domain.ports import AlertLevel
    from src.infrastructure.discord_notification_adapter import DiscordNotificationAdapter

    return DiscordNotificationAdapter().send_alert(
        f"東証の休場日の表に {year} 年の分が無い",
        f"python/src/domain/jp_market_holidays.py に {year} 年の休場日を足す。"
        f"足すまで {year} 年は土日だけを除く判定になり、休場日にも発注し、"
        "足の遅れの線も休場日を営業日に数える。",
        level=AlertLevel.WARNING,
    )


def notify_jp_holiday_table_if_needed(today: date) -> None:
    """次の年の表が要る日に、まだ送っていなければ知らせる。例外は外へ出さない。"""
    try:
        year = missing_jp_holiday_table_year(today)
        if year is None or not _notice_lock.acquire(blocking=False):
            return
        try:
            if _read_notified_day() != today.isoformat() and _send_notice(year):
                _mark_notified(today)
        finally:
            _notice_lock.release()
    except Exception as e:
        logger.error("東証の休場日の表の知らせに失敗: %s", e, exc_info=True)
