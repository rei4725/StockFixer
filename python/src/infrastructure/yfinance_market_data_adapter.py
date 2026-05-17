"""YFinance 市場データアダプター

MarketDataPort ポートの yfinance 実装。
既存の YFinanceDataSource に委譲する。
"""

import pandas as pd

from src.domain.ports import MarketDataPort
from src.market_data.yfinance_source import YFinanceDataSource
from src.utils.logger import get_logger

logger = get_logger(__name__)


class YFinanceMarketDataAdapter(MarketDataPort):
    """yfinance を使った市場データ取得アダプター"""

    def __init__(self) -> None:
        self._source = YFinanceDataSource()

    def get_stock_data(
        self,
        symbol: str,
        market: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self._source.get_stock_data(symbol, market, start_date, end_date)

    def get_forex_data(
        self,
        pair: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        return self._source.get_forex_data(pair, start_date, end_date)
