"""
yfinance クライアントラッパー

yfinance の API を安全にラップし、以下を一元処理する:
- MultiIndex カラムのフラット化（単一銘柄 yf.download 時に発生）
- タイムゾーン情報の除去（DuckDB への保存で tz-naive が必要）
- 全 NaN 行の除去
- リトライ・指数バックオフ（レート制限・ネットワークエラー対応）

このモジュールを経由することで、呼び出し元は正規化済みの DataFrame のみを受け取り、
yfinance のバージョン差異を意識しない。

使用例:
    from src.utils import yf_client

    # period 指定（翌営業日始値確認など）
    df = yf_client.download("7203.T", period="2d")

    # 日付範囲指定
    df = yf_client.download("AAPL", start="2024-01-01", end="2024-12-31")

    # Ticker.history() 方式（並列取得時のスレッドセーフ版）
    df = yf_client.ticker_history("7203.T", start="2024-01-01", end="2024-12-31")
"""

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger
from src.utils.retry_helper import with_retry

logger = get_logger(__name__)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    yf.download() / Ticker.history() が返す DataFrame を正規化する。

    処理内容:
      - MultiIndex カラム → 第1レベル（フィールド名）のみに落とす
      - 全カラム NaN 行を除去
      - インデックスを timezone-naive に変換（DuckDB 保存対応）
    """
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(how="all")

    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def download(
    ticker: str,
    *,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    progress: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    リトライ対応のダウンロードラッパー。正規化済み DataFrame を返す。

    Args:
        ticker:       ティッカーシンボル（例: "7203.T", "AAPL"）
        period:       取得期間（例: "2d", "1mo"）。start/end と排他
        start:        開始日（YYYY-MM-DD）
        end:          終了日（YYYY-MM-DD、exclusive）
        interval:     足種（デフォルト "1d"）
        auto_adjust:  株式分割・配当調整（デフォルト True）
        progress:     tqdm プログレスバー表示（デフォルト False）

    Returns:
        正規化済み OHLCV DataFrame。取得失敗時は空 DataFrame。
    """
    kw: dict = dict(interval=interval, auto_adjust=auto_adjust, progress=progress, **kwargs)
    if period is not None:
        kw["period"] = period
    else:
        kw["start"] = start
        kw["end"] = end

    def _do() -> pd.DataFrame:
        return yf.download(ticker, **kw)

    try:
        df = with_retry(_do)
        return _normalize(df)
    except Exception as e:
        logger.warning(f"[yf_client] download 失敗 ticker={ticker}: {e}", exc_info=True)
        return pd.DataFrame()


def ticker_history(
    ticker: str,
    *,
    start: str,
    end: str,
    auto_adjust: bool = True,
    timeout: int = 30,
    **kwargs,
) -> pd.DataFrame:
    """
    リトライ対応の Ticker.history ラッパー。正規化済み DataFrame を返す。

    yf.download() と異なりスレッドセーフのため、並列取得時に推奨。

    Args:
        ticker:      ティッカーシンボル
        start:       開始日（YYYY-MM-DD）
        end:         終了日（YYYY-MM-DD）
        auto_adjust: 株式分割・配当調整（デフォルト True）
        timeout:     HTTP タイムアウト（秒、デフォルト 30）

    Returns:
        正規化済み OHLCV DataFrame。取得失敗時は空 DataFrame。
    """
    ticker_obj = yf.Ticker(ticker)

    def _do() -> pd.DataFrame:
        return ticker_obj.history(
            start=start, end=end, auto_adjust=auto_adjust, timeout=timeout, **kwargs
        )

    try:
        df = with_retry(_do)
        return _normalize(df)
    except Exception as e:
        logger.warning(f"[yf_client] ticker_history 失敗 ticker={ticker}: {e}", exc_info=True)
        return pd.DataFrame()
