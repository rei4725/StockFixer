"""
マクロ/イベント特徴量モジュール (R-201)

既存クロスアセット特徴量（VIX / USD/JPY / 米10年債）を補完する追加マクロ指標と
カレンダーベースのイベントフラグを提供する。

追加指標:
    fetch_additional_macro_features()
        - sp500_close    : S&P 500（市場トレンド代理）
        - dxy_close      : ドルインデックス DX-Y.NYB
        - gold_close     : 金先物 GC=F（リスクオフ代理）

    add_macro_derived_features()
        - vix_ma20       : VIX 20日移動平均
        - vix_spike      : VIX > 25 フラグ（急騰警戒域）
        - vix_momentum   : VIX 5日変化率
        - tnx_momentum   : 米10年債利回り 5日変化率

    add_event_flags()
        - is_earnings_season : 決算シーズン（1月-2月, 4月-5月, 7月-8月, 10月-11月）
        - is_fomc_week       : FOMC 開催週（年8回、月・水・水曜実施の週）
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 追加クロスアセットティッカー定義
# ---------------------------------------------------------------------------
_ADDITIONAL_MACRO_TICKERS: dict[str, str] = {
    "sp500_close": "^GSPC",  # S&P 500
    "dxy_close": "DX-Y.NYB",  # ドルインデックス
    "gold_close": "GC=F",  # 金先物
}

# 決算シーズン: 月末決算を翌1〜2ヶ月に報告する慣行から逆算
_EARNINGS_SEASON_MONTHS = frozenset([1, 2, 4, 5, 7, 8, 10, 11])

# FOMC 開催予定週の月/日リスト（2025〜2026年の実際のスケジュール）
# 開催日の年は動的に補完するため月日のみを保持する
_FOMC_MONTH_DAY: list[tuple[int, int]] = [
    (1, 29),
    (3, 19),
    (5, 7),
    (6, 18),
    (7, 30),
    (9, 17),
    (10, 29),
    (12, 10),  # 2025
    (1, 28),
    (3, 18),
    (4, 29),
    (6, 17),
    (7, 29),
    (9, 16),
    (10, 28),
    (12, 9),  # 2026
]


def fetch_additional_macro_features(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    S&P500 / ドルインデックス / 金先物を yfinance から取得する。

    既存の fetch_cross_asset_features()（VIX / USD/JPY / 米10年債）を補完する。

    Args:
        start_date: 取得開始日（YYYY-MM-DD）
        end_date:   取得終了日（YYYY-MM-DD）

    Returns:
        カラム: sp500_close, dxy_close, gold_close（取得できたもの）
        全部失敗時は None
    """
    from src.market_data import yf_client

    frames: list[pd.Series] = []
    end_date_extended = (pd.to_datetime(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")

    for col_name, ticker in _ADDITIONAL_MACRO_TICKERS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = yf_client.ticker_history(ticker, start=start_date, end=end_date_extended)
            if hist is None or hist.empty or "Close" not in hist.columns:
                logger.debug("追加マクロ取得スキップ [%s]: データなし", ticker)
                continue
            close = hist["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.rename(col_name)
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)
            frames.append(close)
        except Exception:
            logger.debug("追加マクロ取得スキップ [%s]", ticker, exc_info=True)

    if not frames:
        return None

    result = pd.concat(frames, axis=1, sort=False)
    result.sort_index(inplace=True)
    return result


def add_macro_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    既存のクロスアセット列から派生マクロ指標を計算して追加する。

    前提列（存在しない場合はスキップ）:
        - vix_close   → vix_ma20, vix_spike, vix_momentum
        - tnx_close   → tnx_momentum

    Args:
        df: 日付インデックス付き DataFrame（vix_close, tnx_close 列を含む想定）

    Returns:
        派生特徴量を追加した DataFrame
    """
    result = df.copy()

    if "vix_close" in result.columns:
        result["vix_ma20"] = result["vix_close"].rolling(20, min_periods=1).mean()
        result["vix_spike"] = (result["vix_close"] > 25).astype(int)
        result["vix_momentum"] = result["vix_close"].pct_change(5)

    if "tnx_close" in result.columns:
        result["tnx_momentum"] = result["tnx_close"].diff(5)

    return result


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    カレンダーベースのイベントフラグを追加する。

    追加列:
        - is_earnings_season (int 0/1): 決算シーズン月（1,2,4,5,7,8,10,11）
        - is_fomc_week (int 0/1):       FOMC 開催週（±3日を対象）

    Args:
        df: 日付インデックス付き DataFrame

    Returns:
        イベントフラグを追加した DataFrame
    """
    result = df.copy()
    idx = result.index

    if not isinstance(idx, pd.DatetimeIndex):
        logger.debug("add_event_flags: DatetimeIndex でないためスキップ")
        return result

    # 決算シーズンフラグ
    result["is_earnings_season"] = idx.month.isin(_EARNINGS_SEASON_MONTHS).astype(int)

    # FOMC 開催週フラグ（開催日 ±3 日を 1 とする）
    fomc_dates: set[pd.Timestamp] = set()
    years_in_index = idx.year.unique()
    for year in years_in_index:
        for month, day in _FOMC_MONTH_DAY:
            try:
                fomc_dates.add(pd.Timestamp(year, month, day))
            except ValueError:
                pass

    def _is_fomc_week(ts: pd.Timestamp) -> int:
        return int(any(abs((ts - fd).days) <= 3 for fd in fomc_dates))

    result["is_fomc_week"] = [_is_fomc_week(ts) for ts in idx]

    return result
