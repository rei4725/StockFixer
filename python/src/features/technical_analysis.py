import pandas as pd
import ta

# テクニカル指標のデフォルトパラメータ
_DEFAULT_TA_PARAMS = {
    "macd_slow": 26,
    "macd_fast": 12,
    "macd_sign": 9,
    "ema_fast": 12,
    "ema_slow": 26,
    "atr_window": 14,
    "rsi_window": 14,
    "bb_window": 20,
    "bb_dev": 2,
    "stoch_window": 14,
    "stoch_smooth": 3,
}


def create_basic_lag_features(
    df: pd.DataFrame, n_lags: int = 5, feature_cols=None, target_horizon: int = 1
):
    """
    指定した数値列（または全数値列）について、過去n日分のラグ特徴量を作成する

    Args:
        df (pd.DataFrame): OHLCVやテクニカル指標などを含むDataFrame（インデックスは日付）
        n_lags (int): ラグ数（過去何日分を特徴量にするか）
        feature_cols (list or None): ラグ付与対象の列名リスト。Noneなら全ての数値列。
        target_horizon (int): 予測ホライズン（何営業日後の変化率を予測するか）。デフォルト=1（翌日）。

    Returns:
        X (pd.DataFrame): 特徴量データ
        y (pd.Series): 予測ターゲット（target_horizon日後の変化率）
    """
    df = df.copy()
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[float, int]).columns
    # pd.concat で一括追加することで DataFrame の断片化警告を回避
    new_cols: dict = {}
    for col in feature_cols:
        for lag in range(1, n_lags + 1):
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    # 予測ターゲットは target_horizon 日後の変化率（リターン）
    new_cols["target"] = (df["Close"].shift(-target_horizon) - df["Close"]) / df["Close"]
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.dropna()
    lag_feature_cols = [f"{col}_lag{lag}" for col in feature_cols for lag in range(1, n_lags + 1)]
    X = df[lag_feature_cols]
    y = df["target"]
    return X, y


def add_technical_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
     DataFrameにテクニカル指標を追加する

    Args:
         df (pd.DataFrame): OHLCVデータを含むDataFrame
         params (dict, optional): テクニカル指標のパラメータ辞書。
             省略時は `_DEFAULT_TA_PARAMS` のデフォルト値を使用。
             例: {"rsi_window": 21, "bb_window": 14}

     Returns:
         pd.DataFrame: テクニカル指標が追加されたDataFrame
    """
    p = {**_DEFAULT_TA_PARAMS, **(params or {})}
    df = df.copy()

    # MultiIndex列をフラット化（yfinanceやCSV読み込み時の対策）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.Index(  # type: ignore[assignment]
            [str(col[0]) if isinstance(col, tuple) else str(col) for col in df.columns]
        )

    # 各OHLCV列が2次元の場合は1次元に変換
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            if hasattr(df[col], "ndim") and df[col].ndim > 1:
                df[col] = df[col].iloc[:, 0]  # type: ignore[call-overload]

    # MACD
    macd = ta.trend.MACD(
        close=df["Close"],
        window_slow=p["macd_slow"],
        window_fast=p["macd_fast"],
        window_sign=p["macd_sign"],
        fillna=True,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # EMA
    df["ema_fast"] = ta.trend.EMAIndicator(
        close=df["Close"], window=p["ema_fast"], fillna=True
    ).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(
        close=df["Close"], window=p["ema_slow"], fillna=True
    ).ema_indicator()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=p["atr_window"], fillna=True
    ).average_true_range()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(
        close=df["Close"], window=p["rsi_window"], fillna=True
    ).rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        close=df["Close"], window=p["bb_window"], window_dev=p["bb_dev"], fillna=True
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=p["stoch_window"],
        smooth_window=p["stoch_smooth"],
        fillna=True,
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # OBV (On Balance Volume)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"], fillna=True
    ).on_balance_volume()

    # 季節性特徴量（DatetimeIndexの場合のみ付与）
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_month_end"] = df.index.is_month_end.astype(int)

    return df


def classify_regime(
    df: pd.DataFrame,
    ema_window: int = 200,
    atr_window: int = 14,
    vol_high_quantile: float = 0.67,
) -> pd.Series:
    """
    市場レジームを「bull（上昇）」「bear（下降）」「range（レンジ）」に分類する。

    判定ロジック:
        1. Close と 200日EMA の大小関係でトレンド方向を判定
        2. ATR の高低でボラティリティを分類し、高ボラ局面を「range」に分類

    Args:
        df:               Close, High, Low 列を含む DataFrame（DatetimeIndex 推奨）
        ema_window:       EMA のウィンドウ幅（デフォルト: 200日）
        atr_window:       ATR 計算に使うウィンドウ幅（デフォルト: 14日）
        vol_high_quantile: ATR がこの分位数以上なら高ボラ → "range"（デフォルト: 0.67）

    Returns:
        pd.Series[str]: 各日に "bull" / "bear" / "range" を付与したシリーズ（元の df と同インデックス）
    """
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=str, index=df.index)

    close = pd.Series(df["Close"].values, index=df.index, dtype=float)

    # EMA（データ数が ema_window 未満のときは実際の長さを使用）
    actual_window = min(ema_window, max(1, len(df) - 1))
    ema: pd.Series = close.ewm(span=actual_window, adjust=False).mean()

    # ATR（High / Low がなければ Close のみで簡易計算）
    if "High" in df.columns and "Low" in df.columns:
        high = pd.Series(df["High"].values, index=df.index, dtype=float)
        low = pd.Series(df["Low"].values, index=df.index, dtype=float)
        tr_parts: list[pd.Series] = [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ]
        tr: pd.Series = pd.concat(tr_parts, axis=1).max(axis=1)
    else:
        tr = close.pct_change().abs()

    atr: pd.Series = tr.rolling(window=atr_window, min_periods=1).mean()
    vol_threshold = float(atr.quantile(vol_high_quantile))

    regime = pd.Series("bear", index=df.index, dtype=str)
    regime[close > ema] = "bull"
    regime[atr >= vol_threshold] = "range"

    return regime
