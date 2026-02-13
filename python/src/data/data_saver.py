"""
データ保存モジュール

生の株価データをCSVファイルに保存する機能を提供する
特徴量生成を含む処理は src.services.data_pipeline を使用すること
"""

import os
from datetime import datetime
from typing import Optional
import pandas as pd

from src.data.data_loader import get_stock_data
from src.utils.csv_io import save_dataframe_to_csv
from src.utils.data_path_utils import get_data_subdir, get_ticker, ensure_dir


def save_raw_stock_data(
    market: str,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    out_dir: str = None  # 省略時はdata_path_utilsのデフォルトを使用
) -> Optional[pd.DataFrame]:
    """
    指定した市場・シンボル・期間の生の株価データを取得し、CSVに保存する。
    特徴量生成は行わない（data層の責務として純粋なデータ保存のみ）
    
    Args:
        market: マーケット識別子 (例: "us", "jp")
        symbol: 銘柄シンボル (例: "AAPL", "7203")
        start_date: 開始日 (省略時は取得可能な最古日)
        end_date: 終了日 (省略時は現在日)
        out_dir: 出力ディレクトリのベースパス
    
    Returns:
        保存したDataFrame、または取得失敗時はNone
    """
    # start_date, end_date自動決定
    if start_date is None or end_date is None:
        try:
            df_all = get_stock_data(market, symbol, "1900-01-01", datetime.now().strftime("%Y-%m-%d"))
            if df_all is None or df_all.empty:
                print(f"{symbol} のデータが取得できませんでした。")
                return None
            start_date = df_all.index.min().strftime("%Y-%m-%d")
        except Exception as e:
            print(f"{symbol} のデータ取得でエラー: {e}")
            return None
        end_date = datetime.now().strftime("%Y-%m-%d")

    # 市場ごとにティッカーを補正
    ticker = get_ticker(market, symbol)

    print(f"データ取得: market={market}, symbol={symbol}, ticker={ticker}, {start_date}～{end_date}")
    df = get_stock_data(market, ticker, start_date, end_date)
    if df is None or df.empty:
        print("データが取得できませんでした。")
        return None

    # サブディレクトリ生成
    sub_dir = get_data_subdir(market, symbol)
    ensure_dir(sub_dir)
    
    # ファイル名生成（raw_YYYY_MM_DD_YYYY_MM_DD.csv）
    fname = f"raw_{start_date.replace('-', '_')}_{end_date.replace('-', '_')}.csv"
    out_path = os.path.join(sub_dir, fname)

    # 保存
    save_dataframe_to_csv(df, out_path)
    print(f"保存完了: {out_path}")
    
    return df


# 後方互換性のため、save_stock_data_with_featuresはservices層からインポートするよう促す
def save_stock_data_with_features(*args, **kwargs):
    """
    非推奨: この関数はservices層に移動しました。
    代わりに from src.services.data_pipeline import save_stock_data_with_features を使用してください。
    """
    import warnings
    warnings.warn(
        "save_stock_data_with_features は src.services.data_pipeline に移動しました。"
        "from src.services.data_pipeline import save_stock_data_with_features を使用してください。",
        DeprecationWarning,
        stacklevel=2
    )
    from src.services.data_pipeline import save_stock_data_with_features as _save
    return _save(*args, **kwargs)
