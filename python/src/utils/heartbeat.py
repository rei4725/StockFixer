"""
healthchecks.io 死活監視 ping ユーティリティ（#496）

外部の healthchecks.io へ fire-and-forget で ping を送り、
単一PC・Docker 停止をシステムの外側から検知できるようにする。

- `HEALTHCHECKS_PING_KEY` 未設定（空文字）の場合は完全に no-op
- 送信失敗は warning ログのみ — 監視のためにジョブ本体を落とさない
- slug ベース ping + `?create=1` により check は初回 ping で自動作成される
"""

from __future__ import annotations

import requests

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 監視 ping はジョブ本体を待たせないよう短めに切る
_TIMEOUT_SECONDS = 10

# recovery_poller が5分間隔で送るスケジューラ死活 ping の slug
SCHEDULER_ALIVE_SLUG = "scheduler-alive"


def ping_heartbeat(slug: str, success: bool = True) -> bool:
    """healthchecks.io の check に ping を送る。

    Args:
        slug: check の slug（例: "scheduler-alive", "daily_pipeline"）
        success: False の場合 /fail エンドポイントへ送り check を即 down にする

    Returns:
        HTTP 2xx を受け取った場合 True。無効化時・送信失敗時は False。
    """
    ping_key = settings.HEALTHCHECKS_PING_KEY
    if not ping_key:
        return False

    url = f"{settings.HEALTHCHECKS_BASE_URL}/{ping_key}/{slug}"
    if not success:
        url += "/fail"

    try:
        response = requests.get(url, params={"create": "1"}, timeout=_TIMEOUT_SECONDS)
    except requests.RequestException:
        logger.warning(f"heartbeat ping 送信失敗: slug={slug}", exc_info=True)
        return False

    if 200 <= response.status_code < 300:
        return True

    logger.warning(f"heartbeat ping 非2xx応答: slug={slug} status={response.status_code}")
    return False
