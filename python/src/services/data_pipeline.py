"""
データパイプラインサービス

データ取得 → 特徴量生成 → DB保存 の一連の処理を統合するサービス層
data層とfeatures層を組み合わせて利用する
"""

import re
from datetime import datetime
from typing import Optional

from src.data.data_loader import get_stock_data
from src.data.data_saver import save_raw_stock_data
from src.features.technical_analysis import create_basic_lag_features, add_technical_indicators
from src.utils.db import upsert_stock_features, delete_stock_features
from src.utils.data_path_utils import get_ticker


def save_stock_data_with_features(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_dir: str = None  # 後方互換のため残置（未使用）
):
    """
    指定した市場・シンボル・期間の株価データを取得し、特徴量生成後、DBに保存する。
    
    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (省略時は取得可能な最古日)
        end_date: 終了日 (省略時は現在日)
        out_dir: 後方互換のため残置（未使用）
    """
    # start_date, end_date自動決定
    if start_date is None or end_date is None:
        try:
            df_all = get_stock_data(market, symbol, "1900-01-01", datetime.now().strftime("%Y-%m-%d"))
            if df_all is None or df_all.empty:
                print(f"{symbol} のデータが取得できませんでした。")
                return
            start_date = df_all.index.min().strftime("%Y-%m-%d")
        except Exception as e:
            print(f"{symbol} のデータ取得でエラー: {e}")
            return
        end_date = datetime.now().strftime("%Y-%m-%d")

    # 市場ごとにティッカーを補正
    ticker = get_ticker(market, symbol)

    print(f"データ取得: market={market}, symbol={symbol}, ticker={ticker}, {start_date}～{end_date}")
    df = get_stock_data(market, ticker, start_date, end_date)
    if df is None or df.empty:
        print("データが取得できませんでした。")
        return

    # テクニカル指標を追加
    df = add_technical_indicators(df)

    print("特徴量生成（全数値列ラグ特徴量）...")
    X, y = create_basic_lag_features(df, n_lags=5, feature_cols=None)
    if X is None or X.empty or y is None:
        print("特徴量生成に失敗しました。")
        return

    # 特徴量名の正規化
    def normalize_col(col):
        return re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
    X.columns = [normalize_col(c) for c in X.columns]

    # X, yを1つのDataFrameにまとめる
    data = X.copy()
    
    # market と symbol を列として追加（統合モデル用）
    data['market'] = market
    data['symbol'] = symbol
    # market をエンコード（数値化）
    market_codes = {"us": 0, "jp": 1}
    data['market_encoded'] = market_codes.get(market, -1)
    
    data['y'] = y

    # 既存データを削除してからDBに保存（CSV時代の全削除→書き出しと同等）
    delete_stock_features(market, symbol)
    upsert_stock_features(market, symbol, data)
    print(f"DB保存完了: {market}_{symbol}")
