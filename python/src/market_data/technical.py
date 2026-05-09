import pandas as pd
import ta

from src.market_data.market_regime import get_market_regime

_MULTI_TIMEFRAME_RULES = {
    "weekly": "W-FRI",
    "monthly": "ME",
}

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
    "volume_ma_window": 20,
}


def create_basic_lag_features(
    df: pd.DataFrame, n_lags: int = 10, feature_cols=None, target_horizon: int = 1
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
        feature_cols = [
            c for c in df.select_dtypes(include=[float, int]).columns if c != "earnings_flag"
        ]
    else:
        feature_cols = [c for c in feature_cols if c != "earnings_flag"]
    # pd.concat で一括追加することで DataFrame の断片化警告を回避
    new_cols: dict = {}
    for col in feature_cols:
        for lag in range(1, n_lags + 1):
            new_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    # 予測ターゲットは target_horizon 日後の変化率（リターン）
    new_cols["target"] = (df["Close"].shift(-target_horizon) - df["Close"]) / df["Close"]
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df = df.dropna()
    if "earnings_flag" in df.columns:
        df = df[df["earnings_flag"] == 0]
    lag_feature_cols = [f"{col}_lag{lag}" for col in feature_cols for lag in range(1, n_lags + 1)]
    X = df[lag_feature_cols]
    y = df["target"]
    return X, y


def add_earnings_flag(
    df: pd.DataFrame,
    earnings_dates: pd.DatetimeIndex | list | tuple,
    lookaround_days: int = 3,
) -> pd.DataFrame:
    """決算日前後の営業日に earnings_flag=1 を付与する。"""
    enriched = df.copy()
    enriched["earnings_flag"] = 0

    if not isinstance(enriched.index, pd.DatetimeIndex) or len(enriched) == 0:
        return enriched

    dates = pd.DatetimeIndex(pd.to_datetime(earnings_dates, errors="coerce"))
    if len(dates) == 0:
        return enriched
    if dates.tz is not None:
        dates = dates.tz_localize(None)

    flagged_dates: set[pd.Timestamp] = set()
    for earnings_date in dates.dropna().normalize():
        window = pd.bdate_range(
            earnings_date - pd.offsets.BDay(lookaround_days),
            earnings_date + pd.offsets.BDay(lookaround_days),
        )
        flagged_dates.update(pd.Timestamp(day).normalize() for day in window)

    normalized_index = pd.DatetimeIndex(enriched.index)
    if normalized_index.tz is not None:
        normalized_index = normalized_index.tz_localize(None)
    enriched["earnings_flag"] = normalized_index.normalize().isin(flagged_dates).astype(int)
    return enriched


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    agg_map = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in df.columns:
        agg_map["Volume"] = "sum"

    resampled = df[list(agg_map.keys())].resample(rule).agg(agg_map)  # type: ignore[arg-type]
    return resampled.dropna(subset=["Open", "High", "Low", "Close"])


def _add_multi_timeframe_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return df

    enriched = df.copy()
    for prefix, rule in _MULTI_TIMEFRAME_RULES.items():
        resampled = _resample_ohlcv(df, rule)
        if resampled.empty:
            continue

        tf_df = add_technical_indicators(resampled, params=params, timeframe=prefix)
        tf_features = pd.DataFrame(index=tf_df.index)
        tf_features[f"{prefix}_close_vs_ema_fast"] = tf_df["Close"] / tf_df["ema_fast"] - 1.0
        tf_features[f"{prefix}_close_vs_ema_slow"] = tf_df["Close"] / tf_df["ema_slow"] - 1.0
        tf_features[f"{prefix}_ema_gap"] = tf_df["ema_fast"] / tf_df["ema_slow"] - 1.0
        tf_features[f"{prefix}_rsi"] = tf_df["rsi"]
        tf_features[f"{prefix}_macd_diff"] = tf_df["macd_diff"]

        aligned = tf_features.reindex(enriched.index, method="ffill")
        enriched = pd.concat([enriched, aligned], axis=1)

    return enriched


def add_technical_indicators(
    df: pd.DataFrame, params: dict = None, timeframe: str = "1d"
) -> pd.DataFrame:
    """
     DataFrameにテクニカル指標を追加する

    Args:
         df (pd.DataFrame): OHLCVデータを含むDataFrame
         params (dict, optional): テクニカル指標のパラメータ辞書。
             省略時は `_DEFAULT_TA_PARAMS` のデフォルト値を使用。
             例: {"rsi_window": 21, "bb_window": 14}
         timeframe (str): 入力時間軸。日足(`1d`)時は週足・月足のトレンド特徴量も追加する。

     Returns:
         pd.DataFrame: テクニカル指標が追加されたDataFrame
    """
    p = {**_DEFAULT_TA_PARAMS, **(params or {})}
    df = df.copy()
    row_count = max(len(df), 1)
    macd_slow = max(2, min(int(p["macd_slow"]), row_count))
    macd_fast = max(1, min(int(p["macd_fast"]), macd_slow - 1))
    macd_sign = max(1, min(int(p["macd_sign"]), macd_slow))
    ema_fast_window = max(1, min(int(p["ema_fast"]), row_count))
    ema_slow_window = max(1, min(int(p["ema_slow"]), row_count))
    atr_window = max(1, min(int(p["atr_window"]), row_count))
    rsi_window = max(1, min(int(p["rsi_window"]), row_count))
    bb_window = max(1, min(int(p["bb_window"]), row_count))
    stoch_window = max(1, min(int(p["stoch_window"]), row_count))
    stoch_smooth = max(1, min(int(p["stoch_smooth"]), stoch_window))

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
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_sign,
        fillna=True,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # EMA
    df["ema_fast"] = ta.trend.EMAIndicator(
        close=df["Close"], window=ema_fast_window, fillna=True
    ).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(
        close=df["Close"], window=ema_slow_window, fillna=True
    ).ema_indicator()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=atr_window, fillna=True
    ).average_true_range()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(close=df["Close"], window=rsi_window, fillna=True).rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(
        close=df["Close"], window=bb_window, window_dev=p["bb_dev"], fillna=True
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
        window=stoch_window,
        smooth_window=stoch_smooth,
        fillna=True,
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # OBV (On Balance Volume)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"], fillna=True
    ).on_balance_volume()

    # 出来高プロファイル特徴量（R-213）
    vol_ma_window = max(2, min(int(p["volume_ma_window"]), row_count))
    vol_ma = df["Volume"].rolling(window=vol_ma_window, min_periods=1).mean()
    df["volume_ratio"] = df["Volume"] / vol_ma.replace(0, float("nan"))
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    df["volume_price_trend"] = df["Volume"] * (df["Close"].pct_change().fillna(0))
    df["volume_ma_deviation"] = df["Volume"] / vol_ma.replace(0, float("nan")) - 1.0
    df["volume_ma_deviation"] = df["volume_ma_deviation"].fillna(0.0)

    # 季節性特徴量（DatetimeIndexの場合のみ付与）
    if isinstance(df.index, pd.DatetimeIndex):
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_month_end"] = df.index.is_month_end.astype(int)

    if timeframe == "1d":
        df = _add_multi_timeframe_features(df, p)

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
    return get_market_regime(
        df,
        ema_window=ema_window,
        atr_window=atr_window,
        vol_high_quantile=vol_high_quantile,
    )
