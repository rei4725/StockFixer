"""
クロスアセット・追加マクロの相場の指標を、銘柄の足に結合する。

取得は呼び出し元（pipeline.py）が行い、ここは決まった列の確かめと結合だけを行う。
"""

import threading
from typing import Optional

import pandas as pd

from config.settings import STALE_BARS_MAX_WEEKDAYS
from src.domain.types import MacroStaleSkip
from src.market_data.loader import _CROSS_ASSET_TICKERS
from src.market_data.macro_features import _ADDITIONAL_MACRO_TICKERS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 回の中で列が止まった銘柄（#1634）。足の保存を任されていない銘柄は結果が None になり、
# fetch_stock_data_with_features の戻りでは回の集計へ渡せないので、ここに控える
_stale_skips: dict[tuple[str, str], MacroStaleSkip] = {}
_stale_skips_lock = threading.Lock()


def reset_macro_stale() -> None:
    """回の初めに、前の回の控えを消す。"""
    with _stale_skips_lock:
        _stale_skips.clear()


def take_macro_stale() -> list[MacroStaleSkip]:
    """回の中で列が止まった銘柄を返し、控えを空にする。"""
    with _stale_skips_lock:
        skips = list(_stale_skips.values())
        _stale_skips.clear()
    return skips


def missing_macro_columns(
    cross_asset: Optional[pd.DataFrame], additional_macro: Optional[pd.DataFrame]
) -> list[str]:
    """
    取れた相場の指標の表に無い、決まった列の名前を返す（#1491）。

    表そのものが無い（全部のティッカーが落ちた）ときは、そのグループの列を全部欠けとみなす（#1499）。
    """
    missing: list[str] = []
    for frame, tickers in (
        (cross_asset, _CROSS_ASSET_TICKERS),
        (additional_macro, _ADDITIONAL_MACRO_TICKERS),
    ):
        if frame is None or frame.empty:
            missing += list(tickers)
            continue
        missing += [col for col in tickers if col not in frame.columns]
    return missing


def stale_macro_columns(
    cross_asset: Optional[pd.DataFrame],
    additional_macro: Optional[pd.DataFrame],
    last_bar: pd.Timestamp,
) -> dict[str, str]:
    """
    最後の値の日が銘柄の最後の足の日より線を超えて古い列と、その日を返す（#1622）。

    止まったティッカーの列は表に残り、直近の行が NaN になる。結合の ffill で止まった日の値に
    埋まるので、列の有無では分からない。値が1つも無い列は最後の日を「なし」とする。
    銘柄の足は市場ごとに休場が違うため、休場を数え分けず平日で数える。
    """
    bar_day = _naive_day(last_bar)
    stale: dict[str, str] = {}
    for frame in (cross_asset, additional_macro):
        if frame is None or frame.empty:
            continue  # 欠けた列は missing_macro_columns が見る
        for col in frame.columns:
            last_valid = frame[col].last_valid_index()
            if last_valid is None:
                stale[col] = "なし"
                continue
            day = _naive_day(last_valid)
            behind = len(pd.bdate_range(day + pd.Timedelta(days=1), bar_day))
            if behind > STALE_BARS_MAX_WEEKDAYS:
                stale[col] = day.strftime("%Y-%m-%d")
    return stale


def macro_columns_unusable(
    cross_asset: Optional[pd.DataFrame],
    additional_macro: Optional[pd.DataFrame],
    last_bar: pd.Timestamp,
    market: str,
    symbol: str,
) -> bool:
    """欠けた列か止まった列があれば WARNING を出して True を返す（#1491・#1499・#1622）。"""
    missing = missing_macro_columns(cross_asset, additional_macro)
    if missing:
        # 一部の列だけ取れた銘柄は、欠けた列が予想の道で0に埋まる
        logger.warning(
            f"相場の指標の列が欠けたため特徴量生成をスキップ: {market}/{symbol} "
            f"（欠けた列 {missing}）"
        )
        return True
    stale = stale_macro_columns(cross_asset, additional_macro, last_bar)
    if stale:
        with _stale_skips_lock:
            _stale_skips[(market, symbol)] = MacroStaleSkip(market, symbol, stale)
        # 止まった日の値が前方補完され、予想と発注の VIX に新しい日付の行として入る
        logger.warning(
            f"相場の指標の列が止まったため特徴量生成をスキップ: {market}/{symbol} "
            f"（銘柄の最後の足 {_naive_day(last_bar).strftime('%Y-%m-%d')}、"
            f"平日で {STALE_BARS_MAX_WEEKDAYS}日より古い列と最後の日 {stale}）"
        )
        return True
    return False


def _naive_day(moment: pd.Timestamp) -> pd.Timestamp:
    """タイムゾーンを外した、時刻なしの日付を返す。"""
    day = pd.Timestamp(moment)
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    return day.normalize()


def join_macro_frame(
    df: pd.DataFrame, frame: Optional[pd.DataFrame], label: str, market: str, symbol: str
) -> pd.DataFrame:
    """相場の指標の表を日付で左結合し、その列を前方補完する。表が無ければ df をそのまま返す。"""
    if frame is None or frame.empty:
        return df
    df_idx = df.index
    if isinstance(df_idx, pd.DatetimeIndex) and df_idx.tz is not None:
        df.index = df_idx.tz_localize(None)
    df = df.join(frame, how="left")
    for col in frame.columns:
        if col in df.columns:
            # bfill は未来情報リーク（ルックアヘッド）になるため ffill のみ
            df[col] = df[col].ffill()
    logger.debug("%sを付与: %s (%s/%s)", label, list(frame.columns), market, symbol)
    return df
