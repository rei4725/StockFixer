# coding: utf-8
"""
S&P500全銘柄・Nasdaq10全銘柄リストを取得し、csv追記用データを生成するスクリプト
"""

import pandas as pd

def get_sp500_list():
    # S&P500リストをWikipediaから取得
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    # ティッカー・銘柄名抽出
    return [
        {"市場": "us", "銘柄コード": row["Symbol"].replace(".", "-"), "銘柄名": row["Security"]}
        for _, row in df.iterrows()
    ]

def get_nasdaq100_list():
    # NASDAQ-100全銘柄を取得
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = pd.read_html(url)
    df = tables[4]  # Table 4が['Ticker', 'Company', ...]
    return [
        {"市場": "us", "銘柄コード": row["Ticker"].replace(".", "-"), "銘柄名": row["Company"]}
        for _, row in df.iterrows()
    ]

if __name__ == "__main__":
    sp500 = get_sp500_list()
    nasdaq100 = get_nasdaq100_list()
    # 結果をcsv形式で出力
    for item in sp500:
        print(f'{item["市場"]},{item["銘柄コード"]},{item["銘柄名"]}')
    for item in nasdaq100:
        print(f'{item["市場"]},{item["銘柄コード"]},{item["銘柄名"]}')
