import pandas as pd


def df_to_pretty_string(df: pd.DataFrame, columns=None, header=None) -> str:
    """
    DataFrameから指定カラムのみ抽出し、ヘッダー付きで整形文字列を返す
    """
    if columns is not None:
        df = df[columns]
    s = ""
    if header:
        s += header + "\n"
    s += df.to_string(index=False)
    return s
