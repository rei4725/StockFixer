import pandas as pd
import ta

def create_basic_lag_features(df: pd.DataFrame, n_lags: int = 5, feature_cols=None):
    """
    指定した数値列（または全数値列）について、過去n日分のラグ特徴量を作成する

    Args:
        df (pd.DataFrame): OHLCVやテクニカル指標などを含むDataFrame（インデックスは日付）
        n_lags (int): ラグ数（過去何日分を特徴量にするか）
        feature_cols (list or None): ラグ付与対象の列名リスト。Noneなら全ての数値列。

    Returns:
        X (pd.DataFrame): 特徴量データ
        y (pd.Series): 予測ターゲット（翌日のClose）
    """
    df = df.copy()
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[float, int]).columns
    for col in feature_cols:
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    # 予測ターゲットは翌日の変化率（リターン）
    df['target'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
    df = df.dropna()
    lag_feature_cols = [f"{col}_lag{lag}" for col in feature_cols for lag in range(1, n_lags + 1)]
    X = df[lag_feature_cols]
    y = df['target']
    return X, y

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrameにテクニカル指標を追加する

   Args:
        df (pd.DataFrame): OHLCVデータを含むDataFrame

    Returns:
        pd.DataFrame: テクニカル指標が追加されたDataFrame
    """
    df = df.copy()
    
    # MultiIndex列をフラット化（yfinanceやCSV読み込み時の対策）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in df.columns]
    
    # 各OHLCV列が2次元の場合は1次元に変換
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            if hasattr(df[col], 'ndim') and df[col].ndim > 1:
                df[col] = df[col].iloc[:, 0]

    # MACD
    macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # EMA
    df['ema_fast'] = ta.trend.EMAIndicator(close=df['Close'], window=12, fillna=True).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(close=df['Close'], window=26, fillna=True).ema_indicator()

    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=True).average_true_range()

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14, fillna=True).rsi()

    return df
