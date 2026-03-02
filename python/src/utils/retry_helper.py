"""
リトライロジック・レート制限対応ユーティリティ

yfinanceのAPI制限やネットワークエラーに対応する。
指数バックオフで待機時間を徐々に増やしながらリトライする。
"""

import logging
import time
from typing import Any, Callable, TypeVar

import yfinance as yf

T = TypeVar("T")

# ロギング設定
logger = logging.getLogger(__name__)


def with_retry(
    func: Callable[..., T],
    max_retries: int = 5,
    base_wait_seconds: float = 2.0,
    max_wait_seconds: float = 60.0,
) -> T:
    """
    指数バックオフを使用してyfinance関数をリトライする。

    Args:
        func: 実行する関数（yfinance呼び出し）
        max_retries: 最大リトライ回数（デフォルト5）
        base_wait_seconds: 初期待機時間（秒）（デフォルト2）
        max_wait_seconds: 最大待機時間（秒）（デフォルト60）

    Returns:
        関数の戻り値

    Raises:
        Exception: 全リトライが失敗した場合

    例:
        df = with_retry(lambda: yf.Ticker("AAPL").history(...))
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()

            # レート制限・ネットワークエラーか判定
            is_rate_limit = any(
                keyword in error_msg
                for keyword in [
                    "rate limit",
                    "too many requests",
                    "429",
                    "430",
                    "retry",
                    "temporarily unavailable",
                    "timeout",
                ]
            )
            is_network = any(
                keyword in error_msg
                for keyword in [
                    "connection",
                    "timeout",
                    "reset",
                    "connection refused",
                    "temporarily unavailable",
                ]
            )
            is_retriable = is_rate_limit or is_network

            # 最後のリトライ or リトライ対象外エラーは即 raise
            if attempt == max_retries or not is_retriable:
                logger.error(f"[リトライ失敗] ({attempt}/{max_retries}) {type(e).__name__}: {e}")
                raise

            # 指数バックオフ：wait_time = base_wait * (2 ^ (attempt-1))
            wait_time = min(base_wait_seconds * (2 ** (attempt - 1)), max_wait_seconds)

            if is_rate_limit:
                logger.warning(
                    f"[レート制限] 検出されました。{wait_time:.1f}秒待機後にリトライします... " f"({attempt}/{max_retries})"
                )
            else:
                logger.warning(
                    f"[ネットワークエラー] {wait_time:.1f}秒待機後にリトライします... " f"({attempt}/{max_retries})"
                )

            # カウントダウン表示（5秒以上の場合）
            if wait_time >= 5.0:
                _countdown_wait(wait_time)
            else:
                time.sleep(wait_time)

    # max_retries <= 0 の場合のフォールバック（通常到達しない）
    raise RuntimeError(f"with_retry: max_retries={max_retries} が不正です")


def _countdown_wait(seconds: float):
    """
    カウントダウン表示付きで待機する（5秒以上の場合に使用）。

    Args:
        seconds: 待機時間（秒）
    """
    remaining = int(seconds)
    print("\n待機中: ", end="", flush=True)

    while remaining > 0:
        print(f"{remaining}秒... ", end="", flush=True)
        time.sleep(1)
        remaining -= 1

    print("再試行します。\n", flush=True)


def retry_yfinance_download(
    ticker: str, start: str, end: str, auto_adjust: bool = True, progress: bool = False, **kwargs
) -> Any:
    """
    yf.download() をリトライロジック付きで実行する。

    Args:
        ticker: ティッカーシンボル
        start: 開始日
        end: 終了日
        auto_adjust: 自動調整フラグ
        progress: プログレス表示フラグ
        **kwargs: その他のyf.download()オプション

    Returns:
        DataFrame
    """

    def _download():
        return yf.download(
            ticker, start=start, end=end, auto_adjust=auto_adjust, progress=progress, **kwargs
        )

    return with_retry(_download)


def retry_ticker_history(
    ticker_obj: yf.Ticker,
    start: str,
    end: str,
    auto_adjust: bool = True,
    timeout: int = 30,
    **kwargs,
) -> Any:
    """
    yf.Ticker().history() をリトライロジック付きで実行する。

    Args:
        ticker_obj: yf.Ticket インスタンス
        start: 開始日
        end: 終了日
        auto_adjust: 自動調整フラグ
        timeout: HTTPリクエストタイムアウト（秒）（デフォルト30）
        **kwargs: その他のhistory()オプション

    Returns:
        DataFrame
    """

    def _history():
        return ticker_obj.history(
            start=start, end=end, auto_adjust=auto_adjust, timeout=timeout, **kwargs
        )

    return with_retry(_history)
