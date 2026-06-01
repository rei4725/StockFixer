"""
バックテストパイプラインサービス

バックテストの実行ロジックを統合するサービス層。
run_backtest.py はこのモジュールの関数を呼び出すラッパーとして機能する。
"""

import os
import re
import sys
from datetime import datetime
from typing import Any, Optional, Tuple

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

        from src.market_data.loader import get_stock_data
        from src.market_data.technical import add_technical_indicators, create_basic_lag_features
        from src.utils.data_path_utils import get_ticker

        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        ticker = get_ticker(market, symbol)
        df = get_stock_data(market, ticker, start, end)
        if df is None or df.empty:
            logger.error(f"[backtest] yfinanceからデータを取得できませんでした: {market}/{symbol}")
            sys.exit(1)
        df = df.dropna(axis=1, how="all")
        close_series = df["Close"].copy()
        df = add_technical_indicators(df)
        _nan = int(df.isnull().sum().sum())
        if _nan > 0:
            df = df.ffill().bfill()
        _MIN_ROWS = 30
        if len(df) < _MIN_ROWS:
            logger.error(
                f"[backtest] データ行数不足: {market}/{symbol} "
                f"（{len(df)}行 < 最低{_MIN_ROWS}行）"
            )
            sys.exit(1)
        X, y = create_basic_lag_features(df, n_lags=10)
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
        from src.market_data.loader import get_raw_ohlcv_from_db
        from src.market_data.technical import add_technical_indicators, create_basic_lag_features

        df = get_raw_ohlcv_from_db(market, symbol)
        if df is None or df.empty:
            logger.error(
                f"[backtest] market_data_rawにデータがありません: {market}/{symbol}"
                " → run_data_creation.pyを先に実行してください"
            )
            sys.exit(1)
        df = df.dropna(axis=1, how="all")
        close_series = df["Close"].copy()
        df = add_technical_indicators(df)
        _nan = int(df.isnull().sum().sum())
        if _nan > 0:
            df = df.ffill().bfill()
        _MIN_ROWS = 30
        if len(df) < _MIN_ROWS:
            logger.error(
                f"[backtest] データ行数不足: {market}/{symbol} "
                f"（{len(df)}行 < 最低{_MIN_ROWS}行）"
            )
            sys.exit(1)
        X, y = create_basic_lag_features(df, n_lags=10)
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
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    position_sizing: str = "full",
    position_fraction: float = 0.5,
    atr_risk_pct: float = 0.02,
    atr_multiplier: float = 1.0,
    atr_min_fraction: float = 0.1,
    atr_max_fraction: float = 1.0,
    ensemble: bool = False,
    apply_min_change_filter: bool = False,
    exclude_sentiment: bool = False,
) -> Tuple[pd.DataFrame, dict[str, Any], pd.Series]:
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
        stop_loss_pct: ストップロス率（例: 0.05=5%下落で損切り）
        take_profit_pct: テイクプロフィット率（例: 0.10=10%上昇で利確）
        position_sizing: ポジションサイジング ("full", "fixed", "confidence", "atr")
        position_fraction: 固定ポジション比率（fixed モード用）
        atr_risk_pct: ATRモード: 1トレードあたりのリスク割合（デフォルト: 2%）
        atr_multiplier: ATRモード: ストップ幅のATR倍数（デフォルト: 1.0）
        atr_min_fraction: ATRモード: 建玉下限比率（デフォルト: 10%）
        atr_max_fraction: ATRモード: 建玉上限比率（デフォルト: 100%）
        ensemble: XGBoost+LightGBMアンサンブル予測を使用
        exclude_sentiment: True の場合、センチメント列を特徴量から除外して学習する

    Returns:
        (result_df, metrics, None) のタプル
    """
    from src.backtest.backtester import Backtester
    from src.prediction.manager import ModelManager
    from src.utils.signal_generator import SignalGenerator

    task = _build_task(task_name, threshold)
    model_name = model_name or f"Backtest{model_type}"

    df = load_features(market, symbol, source)

    # 日付列をインデックスに設定（大文字・小文字どちらも対応）
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is not None:
        df = df.set_index(date_col)
        df.index = pd.to_datetime(df.index)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # インデックスが日付型でない場合、期間フィルタはスキップ
        start_date = None
        end_date = None

    # 期間フィルタ
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    # train / test 分割
    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    logger.info(f"学習期間: {train_df.index[0]} ～ {train_df.index[-1]} ({len(train_df)}行)")
    logger.info(f"検証期間: {test_df.index[0]} ～ {test_df.index[-1]} ({len(test_df)}行)")

    # ラベル付与
    label_col = task.label_col
    train_df = train_df.copy()
    test_df = test_df.copy()
    exclude_cols = {label_col, "market", "symbol", "market_encoded"}

    if label_col not in train_df.columns:
        train_df[label_col] = task.make_labels(
            train_df.rename(columns={"close": "Close"}, errors="ignore")
        )
        test_df[label_col] = task.make_labels(
            test_df.rename(columns={"close": "Close"}, errors="ignore")
        )

    _SENTIMENT_PREFIXES = ("sentiment_", "news_count")
    feature_cols = [
        c
        for c in train_df.columns
        if c not in exclude_cols
        and c not in ("Close", "close")
        and not (exclude_sentiment and (c.startswith(_SENTIMENT_PREFIXES)))
    ]

    # NULL を含む行を極力除去しない（90% 以上、有効な特徴量が必要）
    # thresh: 90% の列に値があることを要求
    min_valid = int(len(feature_cols) * 0.9)
    X_train = train_df[feature_cols].dropna(thresh=min_valid)
    y_train = train_df.loc[X_train.index, label_col].dropna()
    X_train = X_train.loc[y_train.index]

    X_test = test_df[feature_cols].dropna(thresh=min_valid)

    # モデル学習・予測
    mm = ModelManager()

    if ensemble:
        pred = _ensemble_predict(mm, X_train, y_train, X_test, model_name)
    else:
        mm.create_model(model_type, model_name)
        mm.train_model(model_name, X_train, y_train)
        pred = pd.Series(mm.predict_with_model(model_name, X_test), index=X_test.index)

    signal = task.make_signal(pred)

    # シミュレーション
    backtester = Backtester(
        model_manager=mm,
        signal_generator=SignalGenerator(market=market, symbol=symbol),
        data_loader=None,
        start_date=None,
        end_date=None,
        market=market,
        symbol=symbol,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_sizing=position_sizing,
        position_fraction=position_fraction,
        atr_risk_pct=atr_risk_pct,
        atr_multiplier=atr_multiplier,
        atr_min_fraction=atr_min_fraction,
        atr_max_fraction=atr_max_fraction,
    )
    result_df, metrics = backtester.simulate_trading(
        test_df.loc[X_test.index],
        signal,
        pred=pred,
    )
    close_col = "Close" if "Close" in test_df.columns else "close"
    price_series = test_df[close_col].rename("price")
    return result_df, metrics, price_series


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
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    position_sizing: str = "full",
    position_fraction: float = 0.5,
    atr_risk_pct: float = 0.02,
    atr_multiplier: float = 1.0,
    atr_min_fraction: float = 0.1,
    atr_max_fraction: float = 1.0,
    ensemble: bool = False,
    enable_short: bool = False,
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
        stop_loss_pct: ストップロス率
        take_profit_pct: テイクプロフィット率
        position_sizing: ポジションサイジング ("full", "fixed", "confidence", "atr")
        position_fraction: 固定ポジション比率（fixed モード用）
        atr_risk_pct: ATRモード時に１トレードでリスクする資金の割合（デフォルト: 2%）
        atr_multiplier: ATRの何倒をストップ幅とするか（デフォルト: 1.0）
        atr_min_fraction: ATRモード: 建玉下限比率（デフォルト: 10%）
        atr_max_fraction: ATRモード: 建玉上限比率（デフォルト: 100%）
        ensemble: XGBoost+LightGBMアンサンブル予測を使用

    Returns:
        (None, None, wf_df) のタプル
    """
    from src.backtest.walk_forward import WalkForwardValidator
    from src.prediction.manager import ModelManager
    from src.utils.signal_generator import SignalGenerator

    task = _build_task(task_name, threshold)
    model_name = model_name or f"Backtest{model_type}"

    mm = ModelManager()
    if not ensemble:
        mm.create_model(model_type, model_name)

    wfv = WalkForwardValidator(
        market=market,
        symbol=symbol,
        model_manager=mm,
        signal_generator=SignalGenerator(market=market, symbol=symbol),
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        n_splits=n_splits,
        source=source,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_sizing=position_sizing,
        position_fraction=position_fraction,
        atr_risk_pct=atr_risk_pct,
        atr_multiplier=atr_multiplier,
        atr_min_fraction=atr_min_fraction,
        atr_max_fraction=atr_max_fraction,
        ensemble=ensemble,
        enable_short=enable_short,
    )

    results_df = wfv.run(model_name=model_name, task=task)
    return None, None, results_df


def save_backtest_results(
    result_df: Optional[pd.DataFrame],
    metrics: Optional[dict[str, Any]],
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
    from src.utils.data_path_utils import ensure_dir, get_results_dir

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


def print_backtest_metrics(
    metrics: dict[str, Any], label: str = "", benchmark: Optional[dict[str, Any]] = None
) -> None:
    """
    バックテストメトリクスを標準出力に表示する。

    Args:
        metrics: compute_metrics が返す辞書
        label: ヘッダーラベル
        benchmark: fetch_benchmark_returns が返す辞書（オプション）
    """
    if not metrics:
        return
    print(f"\n{'='*50}")
    if label:
        print(f" {label}")
    print(f"{'='*50}")

    # net（手数料・スリッページ控除後）
    print("  [NET] 手数料・スリッページ控除後")
    print(f"  {'final_cash':20s}: {metrics.get('final_cash')}")
    print(f"  {'total_return':20s}: {metrics.get('total_return')}")
    print(f"  {'sharpe_ratio':20s}: {metrics.get('sharpe_ratio')}")
    print(f"  {'max_drawdown':20s}: {metrics.get('max_drawdown')}")
    print(f"  {'num_trades':20s}: {metrics.get('num_trades')}")
    print(f"  {'win_rate':20s}: {metrics.get('win_rate')}")
    print(f"  {'profit_factor':20s}: {metrics.get('profit_factor')}")
    if "avg_position_fraction" in metrics:
        print(f"  {'avg_position_fraction':20s}: {metrics.get('avg_position_fraction')}")
        print(f"  {'max_position_fraction':20s}: {metrics.get('max_position_fraction')}")
        print(f"  {'avg_position_value':20s}: {metrics.get('avg_position_value')}")
        print(f"  {'atr_fallback_trades':20s}: {metrics.get('atr_fallback_trades')}")

    # gross（比較用: コスト控除前）
    if "gross_total_return" in metrics:
        print(f"{'─'*50}")
        print("  [GROSS] コスト控除前（同一約定数量ベース）")
        print(f"  {'gross_final_cash':20s}: {metrics.get('gross_final_cash')}")
        print(f"  {'gross_total_return':20s}: {metrics.get('gross_total_return')}")
        print(f"  {'gross_sharpe_ratio':20s}: {metrics.get('gross_sharpe_ratio')}")
        print(f"  {'gross_max_drawdown':20s}: {metrics.get('gross_max_drawdown')}")
        print(f"  {'cost_impact_cash':20s}: {metrics.get('cost_impact_cash')}")
        print(f"  {'cost_impact_return':20s}: {metrics.get('cost_impact_return')}")

    if benchmark and benchmark.get("total_return") is not None:
        bm_ret = benchmark["total_return"]
        strategy_ret = metrics.get("total_return", 0.0)
        alpha = strategy_ret - bm_ret
        print(f"{'─'*50}")
        print(
            f"  {'ベンチマーク':20s}: {benchmark['ticker']} ({benchmark['start']} ～ {benchmark['end']})"
        )
        print(f"  {'BM リターン':20s}: {bm_ret:.4%}")
        print(f"  {'アルファ':20s}: {alpha:+.4%}")
    print(f"{'='*50}")


def plot_backtest_chart(
    result_df: pd.DataFrame,
    metrics: Optional[dict[str, Any]],
    market: str,
    symbol: str,
    initial_cash: float,
    file_notifier=None,
) -> None:
    """
    バックテスト結果グラフを保存し、オプションで通知する。

    Args:
        result_df: バックテスト結果 DataFrame
        metrics: メトリクス辞書
        market: マーケット識別子
        symbol: 銘柄シンボル
        initial_cash: 初期資金
        file_notifier: ``(path: str, title: str) -> None`` の呼び出し可能オブジェクト。
            None の場合は通知しない。呼び出し元（orchestration 等）で注入する。
    """
    from src.backtest.metrics import plot_backtest
    from src.utils.data_path_utils import get_results_dir

    out_dir = os.path.join(get_results_dir(), "backtest", f"{market}_{symbol}")
    chart_path = plot_backtest(
        result_df,
        metrics or {},
        out_dir,
        market=market,
        symbol=symbol,
        initial_cash=initial_cash,
    )
    if chart_path:
        logger.info(f"グラフ保存: {chart_path}")
        if file_notifier is not None:
            file_notifier(chart_path, f"{market.upper()}/{symbol} バックテスト結果")


def fetch_benchmark_for_result(
    result_df: pd.DataFrame,
    benchmark_name: str,
    fallback_start: Optional[str] = None,
    fallback_end: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    バックテスト結果 DataFrame からベンチマーク比較データを取得する。

    Args:
        result_df: バックテスト結果 DataFrame（date 列推奨）
        benchmark_name: ベンチマーク識別子 ("n225", "sp500" など)
        fallback_start: result_df に date 列がない場合の開始日
        fallback_end: result_df に date 列がない場合の終了日

    Returns:
        fetch_benchmark_returns が返す辞書、または None
    """
    from src.backtest.metrics import BENCHMARK_TICKERS, fetch_benchmark_returns

    bm_ticker = BENCHMARK_TICKERS.get(benchmark_name, benchmark_name)
    bm_start = str(result_df["date"].min())[:10] if "date" in result_df.columns else fallback_start
    bm_end = str(result_df["date"].max())[:10] if "date" in result_df.columns else fallback_end
    if bm_start and bm_end:
        return fetch_benchmark_returns(bm_ticker, bm_start, bm_end)
    return None


# --- internal helpers ---


def _build_task(task_name: str, threshold: float) -> Any:
    from src.backtest.task import ReturnRegressionTask

    if task_name == "return_regression":
        return ReturnRegressionTask(threshold=threshold)
    raise ValueError(f"未対応のタスク: {task_name}")


def _ensemble_predict(
    mm: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    base_name: str,
) -> pd.Series:
    """アンサンブル予測（平均）を返す（XGBoost + LightGBM）。

    両モデルを同一データで学習し、予測値の平均を取ることで
    バイアスを低減する。

    Args:
        mm: ModelManager インスタンス
        X_train: 学習用特徴量
        y_train: 学習用ラベル
        X_test: 検証用特徴量
        base_name: ベースモデル名

    Returns:
        アンサンブル予測値の Series
    """
    import numpy as np

    xgb_name = f"{base_name}_XGB"
    lgb_name = f"{base_name}_LGB"

    mm.create_model("XGBoostModel", xgb_name)
    mm.create_model("LightGBMModel", lgb_name)

    mm.train_model(xgb_name, X_train, y_train)
    mm.train_model(lgb_name, X_train, y_train)

    pred_xgb = mm.predict_with_model(xgb_name, X_test)
    pred_lgb = mm.predict_with_model(lgb_name, X_test)

    avg = np.mean([pred_xgb, pred_lgb], axis=0)
    print("  [Ensemble] XGBoost + LightGBM 予測値を平均")
    return pd.Series(avg, index=X_test.index)
