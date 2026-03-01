"""
バックテストパイプラインサービス

バックテストの実行ロジックを統合するサービス層。
run_backtest.py はこのモジュールの関数を呼び出すラッパーとして機能する。
"""
import os
import re
import sys
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


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
        from src.data.data_loader import get_stock_data
        from src.features.technical_analysis import add_technical_indicators, create_basic_lag_features
        from datetime import datetime, timedelta
        from src.utils.data_path_utils import get_ticker

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        ticker = get_ticker(market, symbol)
        df = get_stock_data(market, ticker, start, end)
        if df is None or df.empty:
            print(f"[エラー] yfinanceからデータを取得できませんでした: {market}/{symbol}")
            sys.exit(1)
        close_series = df["Close"].copy()
        df = add_technical_indicators(df)
        X, y = create_basic_lag_features(df, n_lags=5)
        if X is None or X.empty:
            print("[エラー] 特徴量生成に失敗しました。")
            sys.exit(1)
        X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]
        X["y"] = y
        # シミュレーション用に Close 列を保持
        X["Close"] = close_series.reindex(X.index)
        return X

    elif source == "raw":
        from src.data.data_loader import get_raw_ohlcv_from_db
        from src.features.technical_analysis import add_technical_indicators, create_basic_lag_features

        df = get_raw_ohlcv_from_db(market, symbol)
        if df is None or df.empty:
            print(f"[エラー] market_data_rawにデータがありません: {market}/{symbol}")
            print("先に run_data_creation.py を実行してください。")
            sys.exit(1)
        close_series = df["Close"].copy()
        df = add_technical_indicators(df)
        X, y = create_basic_lag_features(df, n_lags=5)
        if X is None or X.empty:
            print("[エラー] 特徴量生成に失敗しました。")
            sys.exit(1)
        X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]
        X["y"] = y
        X["Close"] = close_series.reindex(X.index)
        return X

    else:  # source == "file"
        from src.utils.db import load_stock_features
        df = load_stock_features(market, symbol)
        if df is None or df.empty:
            print(f"[エラー] stock_featuresにデータがありません: {market}/{symbol}")
            print("先に run_data_creation.py を実行してください。")
            sys.exit(1)
        # stock_features には Close 列がないため Close_lag1 で代替
        if "Close" not in df.columns and "Close_lag1" in df.columns:
            df = df.copy()
            df["Close"] = df["Close_lag1"]
        return df


def run_backtest_single(
    market: str,
    symbol: str,
    model_type: str = "XGBoostModel",
    model_name: Optional[str] = None,
    task_name: str = "return_regression",
    threshold: float = 0.0,
    source: str = "file",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    train_ratio: float = 0.8,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.0,
) -> Tuple[pd.DataFrame, dict, None]:
    """
    単一学習/検証期間のバックテストを実行する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_type: モデルタイプ ("XGBoostModel" or "LightGBMModel")
        model_name: モデル名 (Noneなら "Backtest{model_type}")
        task_name: タスク名 ("return_regression")
        threshold: シグナル発生の変化率閾値
        source: データソース ("file" or "raw")
        start_date: バックテスト開始日
        end_date: バックテスト終了日
        train_ratio: 学習データ比率
        initial_cash: 初期資金
        fee_rate: 取引手数料率
        slippage: スリッページ

    Returns:
        (result_df, metrics, None) のタプル
    """
    from src.models.model_manager import ModelManager
    from src.strategy.signal_generator import SignalGenerator
    from src.backtest.backtester import Backtester
    from src.backtest.task import ReturnRegressionTask

    task = _build_task(task_name, threshold)
    model_name = model_name or f"Backtest{model_type}"

    df = load_features(market, symbol, source)

    # 期間フィルタ
    if "Date" in df.columns:
        df = df.set_index("Date")
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    # train / test 分割
    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"学習期間: {train_df.index[0]} ～ {train_df.index[-1]} ({len(train_df)}行)")
    print(f"検証期間: {test_df.index[0]} ～ {test_df.index[-1]} ({len(test_df)}行)")

    # ラベル付与
    label_col = task.label_col
    train_df = train_df.copy()
    test_df = test_df.copy()
    exclude_cols = {label_col, "market", "symbol", "market_encoded"}

    if label_col not in train_df.columns:
        train_df[label_col] = task.make_labels(train_df.rename(columns={"close": "Close"}, errors="ignore"))
        test_df[label_col] = task.make_labels(test_df.rename(columns={"close": "Close"}, errors="ignore"))

    feature_cols = [c for c in train_df.columns if c not in exclude_cols and c not in ("Close", "close")]

    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, label_col]
    X_test = test_df[feature_cols].dropna()

    # モデル学習
    mm = ModelManager()
    mm.create_model(model_type, model_name)
    mm.train_model(model_name, X_train, y_train)

    # 予測 → シグナル
    pred = pd.Series(mm.predict_with_model(model_name, X_test), index=X_test.index)
    signal = task.make_signal(pred)

    # シミュレーション
    backtester = Backtester(
        model_manager=mm,
        signal_generator=SignalGenerator(),
        data_loader=None,
        start_date=None,
        end_date=None,
        market=market,
        symbol=symbol,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
    )
    result_df, metrics = backtester.simulate_trading(test_df.loc[X_test.index], signal)
    return result_df, metrics, None


def run_backtest_walk_forward(
    market: str,
    symbol: str,
    model_type: str = "XGBoostModel",
    model_name: Optional[str] = None,
    task_name: str = "return_regression",
    threshold: float = 0.0,
    source: str = "file",
    n_splits: int = 5,
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.0,
) -> Tuple[None, None, pd.DataFrame]:
    """
    Walk-Forward バックテストを実行する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_type: モデルタイプ
        model_name: モデル名 (Noneなら "Backtest{model_type}")
        task_name: タスク名
        threshold: シグナル発生の変化率閾値
        source: データソース ("file", "api", "raw")
        n_splits: Walk-Forward の分割数
        initial_cash: 初期資金
        fee_rate: 取引手数料率
        slippage: スリッページ

    Returns:
        (None, None, wf_df) のタプル
    """
    from src.models.model_manager import ModelManager
    from src.strategy.signal_generator import SignalGenerator
    from src.backtest.walk_forward import WalkForwardValidator

    task = _build_task(task_name, threshold)
    model_name = model_name or f"Backtest{model_type}"

    mm = ModelManager()
    mm.create_model(model_type, model_name)

    wfv = WalkForwardValidator(
        market=market,
        symbol=symbol,
        model_manager=mm,
        signal_generator=SignalGenerator(),
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        n_splits=n_splits,
        source=source,
    )

    results_df = wfv.run(model_name=model_name, task=task)
    return None, None, results_df


def save_backtest_results(
    result_df: Optional[pd.DataFrame],
    metrics: Optional[dict],
    wf_df: Optional[pd.DataFrame],
    market: str,
    symbol: str,
    task_name: str,
) -> None:
    """
    バックテスト結果を CSV に保存する。

    Args:
        result_df: 取引ログ DataFrame（単一期間モード）
        metrics: メトリクス辞書（単一期間モード）
        wf_df: Walk-Forward 結果 DataFrame
        market: マーケット識別子
        symbol: 銘柄シンボル
        task_name: タスク名
    """
    from src.utils.data_path_utils import get_results_dir, ensure_dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(get_results_dir(), "backtest", f"{market}_{symbol}")
    ensure_dir(out_dir)

    if wf_df is not None and not wf_df.empty:
        path = os.path.join(out_dir, f"wf_{task_name}_{ts}.csv")
        wf_df.to_csv(path, index=False)
        print(f"\n結果保存: {path}")
    elif result_df is not None and not result_df.empty:
        path = os.path.join(out_dir, f"trades_{task_name}_{ts}.csv")
        result_df.to_csv(path, index=False)
        print(f"\n取引ログ保存: {path}")


def print_backtest_metrics(metrics: dict, label: str = "") -> None:
    """
    バックテストメトリクスを標準出力に表示する。

    Args:
        metrics: compute_metrics が返す辞書
        label: ヘッダーラベル
    """
    if not metrics:
        return
    print(f"\n{'='*50}")
    if label:
        print(f" {label}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print(f"{'='*50}")


# --- internal helpers ---

def _build_task(task_name: str, threshold: float):
    from src.backtest.task import ReturnRegressionTask
    if task_name == "return_regression":
        return ReturnRegressionTask(threshold=threshold)
    raise ValueError(f"未対応のタスク: {task_name}")
