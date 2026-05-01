# coding: utf-8
"""
S&P500全銘柄・NASDAQ-100全銘柄リストを取得し、csv追記用データを生成するスクリプト
"""

from src.watchlist.ticker_list import get_nasdaq100_list, get_sp500_list

if __name__ == "__main__":
    for item in get_sp500_list():
        print(f'{item["市場"]},{item["銘柄コード"]},{item["銘柄名"]}')
    for item in get_nasdaq100_list():
        print(f'{item["市場"]},{item["銘柄コード"]},{item["銘柄名"]}')
