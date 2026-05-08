"""
Discord通知のレート制限とデデュープ

- 同一通知の重複抑止（TTL: 10分）
- Webhook レート制限対応（5 req/sec）
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEDUPE_TTL_SECONDS: float = 600.0
_RATE_LIMIT_INTERVAL: float = 1.0 / 5


@dataclass
class _DedupeEntry:
    first_seen: float
    suppressed_count: int = 0


class DiscordRateLimiter:
    """Discord webhook のレート制限とデデュープを管理する。"""

    def __init__(
        self,
        ttl: float = _DEDUPE_TTL_SECONDS,
        rate_interval: float = _RATE_LIMIT_INTERVAL,
    ) -> None:
        self._ttl = ttl
        self._rate_interval = rate_interval
        self._cache: dict[str, _DedupeEntry] = {}
        self._cache_lock = Lock()
        self._rate_lock = Lock()
        self._last_send_time: float = 0.0

    @staticmethod
    def _hash_message(message: str) -> str:
        return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]

    def check_and_record(self, message: str) -> tuple[bool, Optional[str]]:
        """
        デデュープチェックを行い、送信可否と事前送信すべきサマリーを返す。

        Returns:
            (should_send, suppression_summary)
            - should_send: True なら通知を送信してよい
            - suppression_summary: 抑止サマリーメッセージ（なければ None）
        """
        key = self._hash_message(message)
        now = time.monotonic()

        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                self._cache[key] = _DedupeEntry(first_seen=now)
                return True, None

            if now - entry.first_seen < self._ttl:
                entry.suppressed_count += 1
                logger.debug(
                    "Discord通知デデュープ抑止: key=%s, suppressed=%d",
                    key,
                    entry.suppressed_count,
                )
                return False, None

            # TTL 切れ: 抑止件数をサマリーとして返しエントリをリセット
            summary: Optional[str] = None
            if entry.suppressed_count > 0:
                summary = f"[通知抑止サマリー] 同一通知を {entry.suppressed_count} 件抑止しました"
                logger.info(
                    "Discord通知抑止サマリー: key=%s, count=%d",
                    key,
                    entry.suppressed_count,
                )
            self._cache[key] = _DedupeEntry(first_seen=now)
            return True, summary

    def apply_rate_limit(self) -> None:
        """送信レート制限を適用する（5 req/sec を超えないよう sleep）。"""
        with self._rate_lock:
            now = time.monotonic()
            wait = self._rate_interval - (now - self._last_send_time)
            if wait > 0:
                time.sleep(wait)
            self._last_send_time = time.monotonic()


_limiter = DiscordRateLimiter()


def check_and_record(message: str) -> tuple[bool, Optional[str]]:
    """モジュールレベルのデデュープチェック。"""
    return _limiter.check_and_record(message)


def apply_rate_limit() -> None:
    """モジュールレベルのレート制限適用。"""
    _limiter.apply_rate_limit()
