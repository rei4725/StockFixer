"""
kabu STATION® API の流量の制限の断りを受けたときに、待って送り直す。

制限の値（発注系は毎秒5件ほど、余力・情報系は毎秒10件ほど）と断りの応答
（HTTP 429）は、Issue #1442 の見立てによる。公式の仕様書ではまだ確かめていない。
値はこのファイルの定数だけに置く。
"""

import time
from collections.abc import Callable

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

RATE_LIMITED_STATUS = 429  # 流量の制限の断り（見立て）
RATE_LIMIT_MAX_RETRIES = 5  # 断りを受けたときに送り直す最大の回数
RATE_LIMIT_WAIT_SECONDS = 1.0  # 1回目の待ち（秒）。回ごとに足していく

# テストで差し替えるため、モジュールの属性として持つ
_sleep: Callable[[float], None] = time.sleep


def request_with_rate_limit_retry(call: Callable[[], httpx.Response], label: str) -> httpx.Response:
    """HTTP を1回送る関数 call を呼び、流量の制限の断りなら待って送り直す。

    断りでない応答（成功・400・500 など）はそのまま返す。
    送り直しの回数を使い切ったら、最後の断りの応答を返す（呼び手の
    raise_for_status() が例外にする）。
    """
    resp = call()
    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        if resp.status_code != RATE_LIMITED_STATUS:
            return resp
        wait = RATE_LIMIT_WAIT_SECONDS * attempt
        logger.warning(
            "[kabu] 流量の制限の断りを受けた: %s。%.1f 秒待って送り直す (%d/%d)",
            label,
            wait,
            attempt,
            RATE_LIMIT_MAX_RETRIES,
        )
        _sleep(wait)
        resp = call()
    if resp.status_code == RATE_LIMITED_STATUS:
        logger.error(
            "[kabu] 流量の制限の断りが %d 回の送り直しの後も続いた: %s",
            RATE_LIMIT_MAX_RETRIES,
            label,
        )
    return resp
