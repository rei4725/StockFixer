"""
データパイプラインサービス

データ取得 → 特徴量生成 → 保存 の一連の処理を統合するサービス層
data層とfeatures層を組み合わせて利用する
"""

import os
import re
from datetime import datetime
from typing import Optional

from src.data.data_loader import get_stock_data
from src.data.data_saver import save_raw_stock_data
from src.features.technical_analysis import create_basic_lag_features, add_technical_indicators
from src.utils.csv_io import save_dataframe_to_csv
from src.utils.data_path_utils import get_data_dir, get_data_subdir, get_ticker, ensure_dir


def save_stock_data_with_features(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_dir: str = None
):
    """
    指定した市場・シンボル・期間の株価データを取得し、特徴量生成後、out_dirにCSV保存する。
    ファイル名: features_[start_date]_[end_date].csv
    
    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (省略時は取得可能な最古日)
        end_date: 終了日 (省略時は現在日)
        out_dir: 出力ディレクトリのベースパス
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

    # サブディレクトリ生成（out_dirが指定されていなければデフォルトを使用）
    if out_dir is None:
        sub_dir = get_data_subdir(market, symbol)
    else:
        sub_dir = os.path.join(out_dir, f"{market}_{symbol}")
    ensure_dir(sub_dir)
    
    # 既存のcsvファイルを削除
    for file in os.listdir(sub_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(sub_dir, file)
            print(f"既存ファイル削除: {file_path}")
            os.remove(file_path)
    
    # ファイル名生成（features_YYYY_MM_DD_YYYY_MM_DD.csv）
    fname = f"features_{start_date.replace('-', '_')}_{end_date.replace('-', '_')}.csv"
    out_path = os.path.join(sub_dir, fname)

    # 保存
    save_dataframe_to_csv(data, out_path)
    print(f"保存完了: {out_path}")
