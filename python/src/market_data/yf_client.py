"""
yfinance クライアントラッパー

yfinance の API を安全にラップし、以下を一元処理する:
- MultiIndex カラムのフラット化（単一銘柄 yf.download 時に発生）
- タイムゾーン情報の除去（DuckDB への保存で tz-naive が必要）
- 全 NaN 行の除去
- リトライ・指数バックオフ（レート制限・ネットワークエラー対応）
- in-memory + disk キャッシュ（同日中の重複 fetch 回避）

このモジュールを経由することで、呼び出し元は正規化済みの DataFrame のみを受け取り、
yfinance のバージョン差異を意識しない。

使用例:
    from src.market_data import yf_client

    # period 指定（翌営業日始値確認など）
    df = yf_client.download("7203.T", period="2d")

    # 日付範囲指定
    df = yf_client.download("AAPL", start="2024-01-01", end="2024-12-31")

    # Ticker.history() 方式（並列取得時のスレッドセーフ版）
    df = yf_client.ticker_history("7203.T", start="2024-01-01", end="2024-12-31")

    # キャッシュ無効化（--no-cache フラグ対応）
    yf_client.set_no_cache(True)
"""

import pickle
import re
import threading
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.data_path_utils import get_data_dir
from src.utils.logger import get_logger
from src.utils.retry_helper import with_retry

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# キャッシュ層（同一日重複 fetch 回避、#149）
# ---------------------------------------------------------------------------
_TTL_SHORT: float = 15 * 60       # 15分: 当日以降のデータ（未確定）
_TTL_LONG: float = 7 * 24 * 3600  # 7日: 過去データ（終値確定済み）

_memory_cache: dict[str, tuple[pd.DataFrame, float]] = {}
_cache_lock = threading.Lock()
_no_cache: bool = False


def set_no_cache(flag: bool = True) -> None:
    """キャッシュを無効化/有効化する（--no-cache フラグ対応）。"""
    global _no_cache
    _no_cache = flag


def clear_cache() -> None:
    """インメモリ + ディスクキャッシュを全削除する。"""
    global _memory_cache
    with _cache_lock:
        _memory_cache = {}
    cache_dir = Path(get_data_dir()) / "yf_cache"
    if cache_dir.exists():
        for f in cache_dir.glob("*.pkl"):
            try:
                f.unlink()
            except Exception:
                pass


def _cache_key(ticker: str, start: str | None, end: str | None, interval: str) -> str:
    return f"{ticker}|{start}|{end}|{interval}"


def _cache_ttl(end: str | None) -> float:
    """end が今日より前なら長い TTL、当日以降なら短い TTL を返す。"""
    if end is None:
        return _TTL_SHORT
    try:
        end_dt = pd.to_datetime(end).normalize()
        today = pd.Timestamp.now().normalize()
        if end_dt < today:
            return _TTL_LONG
    except Exception:
        pass
    return _TTL_SHORT


def _get_cache_path(key: str) -> Path:
    cache_dir = Path(get_data_dir()) / "yf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^a-zA-Z0-9._-]", "_", key)
    return cache_dir / f"{safe_key}.pkl"


def _cache_get(key: str) -> pd.DataFrame | None:
    now = time.time()
    with _cache_lock:
        entry = _memory_cache.get(key)
    if entry is not None:
        df, expiry = entry
        if now < expiry:
            logger.debug("[yf_cache] memory hit key=%s", key)
            return df
        with _cache_lock:
            _memory_cache.pop(key, None)
    # disk fallback
    path = _get_cache_path(key)
    if path.exists():
        try:
            with open(path, "rb") as fh:
                df, expiry = pickle.load(fh)
            if now < expiry:
                with _cache_lock:
                    _memory_cache[key] = (df, expiry)
                logger.debug("[yf_cache] disk hit key=%s", key)
                return df
            path.unlink(missing_ok=True)
        except Exception:
            pass
    return None


def _cache_set(key: str, df: pd.DataFrame, ttl: float) -> None:
    expiry = time.time() + ttl
    with _cache_lock:
        _memory_cache[key] = (df, expiry)
    path = _get_cache_path(key)
    try:
        with open(path, "wb") as fh:
            pickle.dump((df, expiry), fh)
    except Exception:
        pass  # ディスクキャッシュはベストエフォート


# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
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
    # period 指定は相対時間なのでキャッシュ対象外
    cache_key: str | None = None
    if not _no_cache and period is None and start is not None:
        cache_key = _cache_key(ticker, start, end, interval)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

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
        result = _normalize(df)
    except Exception:
        logger.warning("[yf_client] download 失敗 ticker=%s", ticker, exc_info=True)
        return pd.DataFrame()

    if cache_key is not None and not result.empty:
        _cache_set(cache_key, result, _cache_ttl(end))

    return result


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
    interval = str(kwargs.get("interval", "1d"))
    cache_key: str | None = None
    if not _no_cache:
        cache_key = _cache_key(ticker, start, end, interval)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    ticker_obj = yf.Ticker(ticker)

    def _do() -> pd.DataFrame:
        return ticker_obj.history(
            start=start, end=end, auto_adjust=auto_adjust, timeout=timeout, **kwargs
        )

    try:
        df = with_retry(_do)
        result = _normalize(df)
    except Exception:
        logger.warning("[yf_client] ticker_history 失敗 ticker=%s", ticker, exc_info=True)
        return pd.DataFrame()

    if cache_key is not None and not result.empty:
        _cache_set(cache_key, result, _cache_ttl(end))

    return result
