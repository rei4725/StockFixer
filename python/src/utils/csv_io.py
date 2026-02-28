"""
csv_io.py - CSV入出力ユーティリティ（非推奨）

注意: 本モジュールは後方互換のために残置されています。
新規コードではDuckDB経由のデータアクセス（src.utils.db）を使用してください。
"""

import pandas as pd

def save_dataframe_to_csv(df: pd.DataFrame, path: str, **kwargs):
    """
    DataFrameをCSVファイルとして保存する。
    :param df: 保存するDataFrame
    :param path: 保存先パス
    :param kwargs: pandas.DataFrame.to_csvの追加引数
    """
    try:
        df.to_csv(path, index=False, **kwargs)
        print(f"DataFrameをCSVファイル '{path}' に保存しました。")
    except Exception as e:
        print(f"CSV保存時にエラーが発生しました: {e}")

def load_dataframe_from_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    CSVファイルをDataFrameとして読み込む。
    :param path: 読み込み元パス
    :param kwargs: pandas.read_csvの追加引数
    :return: 読み込んだDataFrame
    """
    try:
        df = pd.read_csv(path, **kwargs)
        print(f"CSVファイル '{path}' からDataFrameを読み込みました。")
        return df
    except Exception as e:
        print(f"CSV読み込み時にエラーが発生しました: {e}")
        return None
