"""
クロスアセット・追加マクロの相場の指標を、銘柄の足に結合する。

取得は呼び出し元（pipeline.py）が行い、ここは決まった列の確かめと結合だけを行う。
"""

from typing import Optional

import pandas as pd

from src.market_data.loader import _CROSS_ASSET_TICKERS
from src.market_data.macro_features import _ADDITIONAL_MACRO_TICKERS
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
