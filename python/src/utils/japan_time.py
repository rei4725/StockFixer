"""日本時間の日時取得・整形ユーティリティ。"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

JAPAN_TIMEZONE = ZoneInfo("Asia/Tokyo")
DEFAULT_JA_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S JST"
DEFAULT_UTC_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def now_jst() -> datetime:
    """現在の日本時間を返す。"""
    return datetime.now(JAPAN_TIMEZONE)


def now_utc() -> datetime:
    """現在の UTC 時刻を返す。"""
    return datetime.now(timezone.utc)


def to_jst(value: datetime) -> datetime:
    """任意の日時を日本時間へ正規化する。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=JAPAN_TIMEZONE)
    return value.astimezone(JAPAN_TIMEZONE)


def to_utc(value: datetime) -> datetime:
    """任意の日時を UTC へ正規化する。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def format_jst(value: datetime | None = None, fmt: str = DEFAULT_JA_DATETIME_FORMAT) -> str:
    """日本時間の書式済み文字列を返す。"""
    target = now_jst() if value is None else to_jst(value)
    return target.strftime(fmt)


def isoformat_jst(value: datetime | None = None) -> str:
    """日本時間の ISO 8601 文字列を返す。"""
    target = now_jst() if value is None else to_jst(value)
    return target.isoformat(timespec="seconds")


def isoformat_utc(value: datetime | None = None) -> str:
    """UTC の ISO 8601 文字列を返す。"""
    target = now_utc() if value is None else to_utc(value)
    return target.isoformat(timespec="seconds")


def parse_iso_datetime(value: str) -> datetime:
    """ISO 8601 文字列を datetime に変換する。"""
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_jst_from_iso(
    value: str | None, fmt: str = DEFAULT_JA_DATETIME_FORMAT, fallback: str = "不明"
) -> str:
    """ISO 8601 文字列を JST 表示用文字列へ変換する。"""
    if not value:
        return fallback
    try:
        return format_jst(parse_iso_datetime(value), fmt=fmt)
    except ValueError:
        return value
