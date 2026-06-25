"""バックテスト用の特徴量ロード。

`load_features` はデータソース ("file" / "api" / "raw") に応じて
特徴量 DataFrame を構築する。run_backtest_single など実行系から呼ばれる。
"""

import re
import sys

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_features(market: str, symbol: str, source: str) -> pd.DataFrame:
    """
    特徴量 DataFrame を取得する。

    Args:
        market: マーケット識別子 (例: "jp", "us")
        symbol: 銘柄シンボル (例: "7203", "AAPL")
        source: "file"=stock_features テーブル,
                "api"=yfinanceから直接取得（Close列付き）,
                "raw"=market_data_raw から再生成

    Returns:
        特徴量 DataFrame（インデックス=日付 or row_num）
        - "api" / "raw" の場合は Close 列を保持
        - "file" の場合は Close_lag1 を Close として補完
    """
    if source == "api":
        from datetime import datetime, timedelta

        from src.backtest.data_port import get_backtest_data_port
        from src.utils.data_path_utils import get_ticker

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        ticker = get_ticker(market, symbol)
        _dp = get_backtest_data_port()
        df = _dp.get_stock_data(market, ticker, start, end)
        if df is None or df.empty:
            logger.error(f"[backtest] yfinanceからデータを取得できませんでした: {market}/{symbol}")
            sys.exit(1)
        df = df.dropna(axis=1, how="all")
        close_series = df["Close"].copy()
        df = _dp.add_technical_indicators(df)
        _nan = int(df.isnull().sum().sum())
        if _nan > 0:
            # 時系列前処理では ffill のみ（過去→現在方向の補完）。
            # bfill は未来情報リーク（ルックアヘッド）になるため使わず、
            # 先頭の埋められない NaN 行は dropna で除去する。
            df = df.ffill().dropna()
        _MIN_ROWS = 30
        if len(df) < _MIN_ROWS:
            logger.error(
                f"[backtest] データ行数不足: {market}/{symbol} "
                f"（{len(df)}行 < 最低{_MIN_ROWS}行）"
            )
            sys.exit(1)
        X, y = _dp.create_basic_lag_features(df, n_lags=10)
        if X is None or X.empty:
            nan_cols = df.isnull().sum()
            nan_cols = nan_cols[nan_cols > 0]
            detail = f"NaN残存列: {nan_cols.to_dict()}" if not nan_cols.empty else "NaN列なし"
            logger.error(
                f"[backtest] 特徴量生成に失敗しました: {market}/{symbol}（{len(df)}行, {detail}）"
            )
            sys.exit(1)
        X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]
        X["y"] = y
        # シミュレーション用に Close 列を保持
        X["Close"] = close_series.reindex(X.index)
        if "atr" in df.columns:
            X["atr"] = df["atr"].reindex(X.index)
        return X

    elif source == "raw":
        from src.backtest.data_port import get_backtest_data_port

        _dp = get_backtest_data_port()
        df = _dp.get_raw_ohlcv_from_db(market, symbol)
        if df is None or df.empty:
            logger.error(
                f"[backtest] market_data_rawにデータがありません: {market}/{symbol}"
                " → run_data_creation.pyを先に実行してください"
            )
            sys.exit(1)
        df = df.dropna(axis=1, how="all")
        close_series = df["Close"].copy()
        df = _dp.add_technical_indicators(df)
        _nan = int(df.isnull().sum().sum())
        if _nan > 0:
            # 時系列前処理では ffill のみ（過去→現在方向の補完）。
            # bfill は未来情報リーク（ルックアヘッド）になるため使わず、
            # 先頭の埋められない NaN 行は dropna で除去する。
            df = df.ffill().dropna()
        _MIN_ROWS = 30
        if len(df) < _MIN_ROWS:
            logger.error(
                f"[backtest] データ行数不足: {market}/{symbol} "
                f"（{len(df)}行 < 最低{_MIN_ROWS}行）"
            )
            sys.exit(1)
        X, y = _dp.create_basic_lag_features(df, n_lags=10)
        if X is None or X.empty:
            nan_cols = df.isnull().sum()
            nan_cols = nan_cols[nan_cols > 0]
            detail = f"NaN残存列: {nan_cols.to_dict()}" if not nan_cols.empty else "NaN列なし"
            logger.error(
                f"[backtest] 特徴量生成に失敗しました: {market}/{symbol}（{len(df)}行, {detail}）"
            )
            sys.exit(1)
        X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]
        X["y"] = y
        X["Close"] = close_series.reindex(X.index)
        if "atr" in df.columns:
            X["atr"] = df["atr"].reindex(X.index)
        return X

    else:  # source == "file"
        from src.utils.db import load_stock_features

        df = load_stock_features(market, symbol)
        if df is None or df.empty:
            logger.error(
                f"[backtest] stock_featuresにデータがありません: {market}/{symbol}"
                " → run_data_creation.pyを先に実行してください"
            )
            sys.exit(1)

        # 100% NULL の列を除去（Dividends、Capital_Gains、Stock_Splits など）
        df = df.dropna(axis=1, how="all")

        # stock_features には Close 列がないため Close_lag1 で代替
        if "Close" not in df.columns and "Close_lag1" in df.columns:
            df = df.copy()
            df["Close"] = df["Close_lag1"]
        if "atr" not in df.columns and "atr_lag1" in df.columns:
            df = df.copy()
            df["atr"] = df["atr_lag1"]

        # date 列があれば DatetimeIndex に復元する
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df.index.name = "Date"
        else:
            # 旧データ（date列なし）: market_data_raw の ts 列で補完を試みる
            try:
                from src.utils.db._connection import _db_connection

                with _db_connection() as con:
                    raw_dates = con.execute(
                        "SELECT ts FROM market_data_raw "
                        "WHERE market = ? AND symbol = ? AND timeframe = 'daily' "
                        "ORDER BY ts",
                        [market, symbol],
                    ).fetchdf()
                if not raw_dates.empty:
                    ts_series = pd.to_datetime(raw_dates["ts"])
                    # create_basic_lag_features(n_lags=10, target=1) で先頭10行・末尾1行が除去される
                    # 位置合わせ: raw_dates[10:-1] が stock_features の行数と一致する想定
                    n_lags_assumed = 10
                    expected_start = n_lags_assumed
                    expected_end = len(ts_series) - 1  # target shift -1
                    aligned = ts_series.iloc[expected_start:expected_end].values
                    if len(aligned) == len(df):
                        df = df.copy()
                        df.index = pd.DatetimeIndex(aligned, name="Date")
                        logger.debug(
                            f"[backtest] market_data_raw から日付を補完: {market}/{symbol}"
                        )
            except Exception as e:
                logger.debug(f"[backtest] 日付補完スキップ: {e}", exc_info=True)

        return df
