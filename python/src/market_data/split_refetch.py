"""
差分の足に株式分割があったときの全期間の取り直し（#817）

yfinance は分割の前の足も調整し直して返すが、DB の古い足は分割前の値段のまま。
差分とつなぐと分割の比の段差になるため、全期間を取り直す。
"""

from typing import Callable, Optional

import pandas as pd

from src.market_data.base import DataSourceBase
from src.utils.logger import get_logger

logger = get_logger(__name__)


def stock_split_ratios(fresh_data: Optional[pd.DataFrame]) -> list:
    """差分の足の Stock Splits の列から、0 でない分割の比を返す。列が無ければ空。"""
    if fresh_data is None or "Stock Splits" not in fresh_data.columns:
        return []
    splits = pd.to_numeric(fresh_data["Stock Splits"], errors="coerce").fillna(0.0)
    return [float(v) for v in splits[splits != 0]]


def refetch_full_history(
    market: str,
    symbol: str,
    ticker: str,
    start_date: str,
    end_date: str,
    split_ratios: list,
    data_source: Optional[DataSourceBase],
    fetch_stock_data: Callable[[str, str, str, str], Optional[pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    """
    株式分割のため start_date〜end_date の全期間の足を取り直す。

    Returns:
        取り直した全期間の足。取れなければ None（段差のある足で特徴量を作らない）。
    """
    logger.info(
        f"株式分割のため全期間を取り直す: {market}/{symbol} "
        f"（分割の比 {split_ratios}、{start_date}～{end_date}）"
    )
    if data_source is not None:
        full_data = data_source.get_stock_data(symbol, market, start_date, end_date)
    else:
        full_data = fetch_stock_data(market, ticker, start_date, end_date)
    if full_data is None or full_data.empty:
        logger.warning(
            f"株式分割の取り直しに失敗したため特徴量生成をスキップ: {market}/{symbol}"
            "（段差のある足で特徴量を作らない）"
        )
        return None
    return full_data
