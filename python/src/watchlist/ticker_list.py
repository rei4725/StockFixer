# coding: utf-8
"""
S&P500 / NASDAQ-100 銘柄リストを Wikipedia から取得するサービス関数
"""

from typing import List, TypedDict


class TickerInfo(TypedDict):
    市場: str
    銘柄コード: str
    銘柄名: str


def get_sp500_list() -> List[TickerInfo]:
    """Wikipedia から S&P500 全銘柄リストを取得する。"""
    import pandas as pd

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return [
        {"市場": "us", "銘柄コード": row["Symbol"].replace(".", "-"), "銘柄名": row["Security"]}
        for _, row in df.iterrows()
    ]


def get_nasdaq100_list() -> List[TickerInfo]:
    """Wikipedia から NASDAQ-100 全銘柄リストを取得する。"""
    import pandas as pd

    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    df = tables[4]
    return [
        {"市場": "us", "銘柄コード": row["Ticker"].replace(".", "-"), "銘柄名": row["Company"]}
        for _, row in df.iterrows()
    ]
