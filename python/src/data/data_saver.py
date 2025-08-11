import os
from datetime import datetime
from src.data.data_loader import get_stock_data
from src.features.technical_analysis import create_basic_lag_features, add_technical_indicators
from src.utils.csv_io import save_dataframe_to_csv

from typing import Optional

def save_stock_data_with_features(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_dir: str = "python/data"
):
    """
    指定した市場・シンボル・期間の株価データを取得し、特徴量生成後、out_dirにCSV保存する。
    ファイル名: [market]_[symbol]_[start_date]_[end_date].csv
    """
    # start_date, end_date自動決定
    if start_date is None or end_date is None:
        try:
            df_all = get_stock_data(symbol, "1900-01-01", datetime.now().strftime("%Y-%m-%d"))
            if df_all is None or df_all.empty:
                print(f"{symbol} のデータが取得できませんでした。")
                return
            start_date = df_all.index.min().strftime("%Y-%m-%d")
        except Exception as e:
            print(f"{symbol} のデータ取得でエラー: {e}")
            return
        end_date = datetime.now().strftime("%Y-%m-%d")

    # 市場ごとにティッカーを補正
    ticker = symbol
    if market.lower() in ["jp", "japan"]:
        if not symbol.endswith(".T"):
            ticker = f"{symbol}.T"
    elif market.lower() in ["us", "usa", "nyse", "nasdaq"]:
        ticker = symbol  # US株はそのまま
    # 他市場は必要に応じて拡張

    print(f"データ取得: market={market}, symbol={symbol}, ticker={ticker}, {start_date}～{end_date}")
    df = get_stock_data(ticker, start_date, end_date)
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
    import re
    def normalize_col(col):
        return re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
    X.columns = [normalize_col(c) for c in X.columns]

    # X, yを1つのDataFrameにまとめる
    data = X.copy()
    data['y'] = y

    # サブディレクトリ生成
    sub_dir = os.path.join(out_dir, f"{market}_{symbol}")
    os.makedirs(sub_dir, exist_ok=True)
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
