"""バックテスト用の特徴量ロード。

`load_features` はデータソース ("file" / "api" / "raw") に応じて
特徴量 DataFrame を構築する。run_backtest_single など実行系から呼ばれる。
"""

import re

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 約定（翌バー始値 #493）と動的スリッページの平均出来高（#494）のためだけに載せる生の列。
# 当バーの値なので、モデルの特徴量には使わない（呼び出し側で除外する）。
SIMULATION_ONLY_COLUMNS = ("Open", "Volume")


class FeaturesUnavailableError(Exception):
    """バックテスト用の特徴量が用意できない（データが無い・行数不足・生成失敗）。

    バッチでは1銘柄のエラーとして except Exception で受けられるよう、SystemExit にしない（#972）。
    CLI は入口の except Exception で受けて終了コード1で終える。
    """


def _attach_simulation_columns(X: pd.DataFrame, price_df: pd.DataFrame) -> None:
    """生の Open・Volume を X の行（日付）に合わせて足す。"""
    for col in SIMULATION_ONLY_COLUMNS:
        if col in price_df.columns:
            X[col] = price_df[col].reindex(X.index)


def _naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _attach_raw_prices(df: pd.DataFrame, market: str, symbol: str) -> pd.DataFrame:
    """stock_features に market_data_raw の生の Open・Volume・Close を日付で合わせて足す。

    stock_features には生の価格が無い（ラグ列だけ）。取れなければ df をそのまま返し、
    呼び出し側は従来どおり Close_lag1 で代替する（約定は Close・スリッページはフラットのみ）。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(
            f"[backtest] 日付が無いため生の Open・Volume を足せません: {market}/{symbol}"
            " → 約定は Close、スリッページはフラットのみになります"
        )
        return df
    try:
        from src.backtest.data_port import get_backtest_data_port

        raw = get_backtest_data_port().get_raw_ohlcv_from_db(market, symbol)
    except Exception:
        logger.warning(
            f"[backtest] market_data_raw の読み込みに失敗: {market}/{symbol}", exc_info=True
        )
        raw = None
    needed = ["Open", "Close", "Volume"]
    if raw is None or raw.empty or not set(needed) <= set(raw.columns):
        logger.warning(
            f"[backtest] market_data_raw に生の Open・Volume・Close がありません: {market}/{symbol}"
            " → Close は Close_lag1 で代替、約定は Close、スリッページはフラットのみになります"
        )
        return df

    raw = raw[needed].copy()
    raw.index = _naive_dates(raw.index)
    raw = raw[~raw.index.duplicated(keep="last")]
    aligned = raw.reindex(_naive_dates(df.index))
    aligned.index = df.index

    missing = int(aligned["Close"].isna().sum())
    if missing > 0:
        logger.warning(
            f"[backtest] market_data_raw に無い日が {missing}/{len(df)} 行: {market}/{symbol}"
            " → その行の Close は Close_lag1、Open は Close で代替します"
        )

    df = df.copy()
    if "Close" not in df.columns:
        close = aligned["Close"]
        if "Close_lag1" in df.columns:
            close = close.fillna(df["Close_lag1"])
        df["Close"] = close
    if "Open" not in df.columns:
        df["Open"] = aligned["Open"].fillna(df["Close"])
    if "Volume" not in df.columns:
        # 欠けた日は直前の出来高で埋める（過去→現在方向のみ。先読みにしない）
        df["Volume"] = aligned["Volume"].ffill().fillna(0.0)
    return df


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
        - "api" / "raw" の場合は生の Close・Open・Volume 列を保持
        - "file" の場合は market_data_raw から生の Close・Open・Volume を日付で補完
          （取れなければ Close_lag1 を Close として補完）
        - Open・Volume（SIMULATION_ONLY_COLUMNS）は約定とスリッページ用で、特徴量にしない
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
            raise FeaturesUnavailableError(f"yfinanceからデータを取得できません: {market}/{symbol}")
        df = df.dropna(axis=1, how="all")
        close_series = df["Close"].copy()
        price_df = df[[c for c in SIMULATION_ONLY_COLUMNS if c in df.columns]].copy()
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
            raise FeaturesUnavailableError(
                f"データ行数不足: {market}/{symbol}（{len(df)}行 < 最低{_MIN_ROWS}行）"
            )
        X, y = _dp.create_basic_lag_features(df, n_lags=10)
        if X is None or X.empty:
            nan_cols = df.isnull().sum()
            nan_cols = nan_cols[nan_cols > 0]
            detail = f"NaN残存列: {nan_cols.to_dict()}" if not nan_cols.empty else "NaN列なし"
            logger.error(
                f"[backtest] 特徴量生成に失敗しました: {market}/{symbol}（{len(df)}行, {detail}）"
            )
            raise FeaturesUnavailableError(f"特徴量生成に失敗しました: {market}/{symbol}")
        X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]
        X["y"] = y
        # シミュレーション用に Close・Open・Volume 列を保持
        X["Close"] = close_series.reindex(X.index)
        _attach_simulation_columns(X, price_df)
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
            raise FeaturesUnavailableError(
                f"market_data_rawにデータがありません: {market}/{symbol}"
            )
        df = df.dropna(axis=1, how="all")
        close_series = df["Close"].copy()
        price_df = df[[c for c in SIMULATION_ONLY_COLUMNS if c in df.columns]].copy()
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
            raise FeaturesUnavailableError(
                f"データ行数不足: {market}/{symbol}（{len(df)}行 < 最低{_MIN_ROWS}行）"
            )
        X, y = _dp.create_basic_lag_features(df, n_lags=10)
        if X is None or X.empty:
            nan_cols = df.isnull().sum()
            nan_cols = nan_cols[nan_cols > 0]
            detail = f"NaN残存列: {nan_cols.to_dict()}" if not nan_cols.empty else "NaN列なし"
            logger.error(
                f"[backtest] 特徴量生成に失敗しました: {market}/{symbol}（{len(df)}行, {detail}）"
            )
            raise FeaturesUnavailableError(f"特徴量生成に失敗しました: {market}/{symbol}")
        X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]
        X["y"] = y
        X["Close"] = close_series.reindex(X.index)
        _attach_simulation_columns(X, price_df)
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
            raise FeaturesUnavailableError(f"stock_featuresにデータがありません: {market}/{symbol}")

        # 最新日の行は y（翌日変化率）が未確定の NaN で保存される（予測用）
        if "y" in df.columns:
            df = df[df["y"].notna()].copy()

        # 100% NULL の列を除去（Dividends、Capital_Gains、Stock_Splits など）
        df = df.dropna(axis=1, how="all")

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
                    raw_dates = pd.read_sql(
                        "SELECT ts FROM market_data_raw "
                        "WHERE market = %s AND symbol = %s AND timeframe = 'daily' "
                        "ORDER BY ts",
                        con,
                        params=[market, symbol],
                    )
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

        # stock_features には生の Close・Open・Volume が無いため market_data_raw から日付で足す
        df = _attach_raw_prices(df, market, symbol)

        # market_data_raw から取れなければ Close_lag1 で代替（従来の動き）
        if "Close" not in df.columns and "Close_lag1" in df.columns:
            df = df.copy()
            df["Close"] = df["Close_lag1"]
        if "atr" not in df.columns and "atr_lag1" in df.columns:
            df = df.copy()
            df["atr"] = df["atr_lag1"]

        return df
