"""
バックテスト実行スクリプト

推論タスク・モデルの組み合わせをバックテストで検証し、
各種パフォーマンス指標を出力・CSV保存する。

使用例:
    # 単一銘柄・単一期間
    py run_backtest.py --market jp --symbol 7203

    # Walk-Forwardに切り替え
    py run_backtest.py --market jp --symbol 7203 --walk-forward

    # モデル・閾値指定
    py run_backtest.py --market us --symbol AAPL \\
        --model-type LightGBMModel --threshold 0.005

    # 生OHLCVからfeature再生成して実行
    py run_backtest.py --market jp --symbol 7203 --source raw
"""

import argparse
import os
import re
import sys
from datetime import datetime

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="バックテストを実行する")
    parser.add_argument("--market", type=str, default="jp", help="マーケット (例: jp, us)")
    parser.add_argument("--symbol", type=str, required=True, help="銘柄コード (例: 7203, AAPL)")
    parser.add_argument("--start-date", type=str, default=None, help="バックテスト開始日 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="バックテスト終了日 YYYY-MM-DD")
    parser.add_argument(
        "--model-type",
        type=str,
        default="XGBoostModel",
        choices=["XGBoostModel", "LightGBMModel"],
        help="モデルタイプ (default: XGBoostModel)",
    )
    parser.add_argument("--model-name", type=str, default=None, help="モデル名 (default: Backtest{model_type})")
    parser.add_argument(
        "--task",
        type=str,
        default="return_regression",
        choices=["return_regression"],
        help="推論タスク (default: return_regression)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="シグナル発生の変化率閾値 (default: 0.0)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="file",
        choices=["file", "raw"],
        help="データソース: 'file'=DB特徴量, 'raw'=DBのOHLCVから再生成 (default: file)",
    )
    parser.add_argument("--walk-forward", action="store_true", help="Walk-Forward 検証を使用する")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-Forward の分割数 (default: 5)")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="単一期間モードの学習データ比率 (default: 0.8)",
    )
    parser.add_argument("--initial-cash", type=float, default=1_000_000, help="初期資金 (default: 1,000,000)")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="取引手数料率 (default: 0.001)")
    parser.add_argument("--slippage", type=float, default=0.0, help="スリッページ (default: 0.0)")
    return parser.parse_args()


def build_task(task_name: str, threshold: float):
    from src.backtest.task import ReturnRegressionTask
    if task_name == "return_regression":
        return ReturnRegressionTask(threshold=threshold)
    raise ValueError(f"未対応のタスク: {task_name}")


def load_features(market: str, symbol: str, source: str) -> pd.DataFrame:
    """特徴量DataFrameを取得する（DBの stock_features または market_data_raw + 再生成）"""
    if source == "raw":
        from src.data.data_loader import get_raw_ohlcv_from_db
        from src.features.technical_analysis import add_technical_indicators, create_basic_lag_features
        import re as _re

        df = get_raw_ohlcv_from_db(market, symbol)
        if df is None or df.empty:
            print(f"[エラー] market_data_rawにデータがありません: {market}/{symbol}")
            print("先に run_data_creation.py を実行してください。")
            sys.exit(1)
        df = add_technical_indicators(df)
        X, y = create_basic_lag_features(df, n_lags=5)
        if X is None or X.empty:
            print("[エラー] 特徴量生成に失敗しました。")
            sys.exit(1)
        X.columns = [_re.sub(r"[^0-9a-zA-Z_]", "_", str(c)) for c in X.columns]
        X["y"] = y
        return X
    else:
        from src.utils.db import load_stock_features
        df = load_stock_features(market, symbol)
        if df is None or df.empty:
            print(f"[エラー] stock_featuresにデータがありません: {market}/{symbol}")
            print("先に run_data_creation.py を実行してください。")
            sys.exit(1)
        return df


def run_single(args):
    """単一学習/検証期間バックテスト"""
    from src.models.model_manager import ModelManager
    from src.strategy.signal_generator import SignalGenerator
    from src.backtest.backtester import Backtester
    from src.backtest.metrics import compute_metrics

    task = build_task(args.task, args.threshold)
    model_name = args.model_name or f"Backtest{args.model_type}"

    df = load_features(args.market, args.symbol, args.source)

    # 期間フィルタ
    if "Date" in df.columns:
        df = df.set_index("Date")
    if args.start_date:
        df = df[df.index >= args.start_date]
    if args.end_date:
        df = df[df.index <= args.end_date]

    # train / test 分割
    split = int(len(df) * args.train_ratio)
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
        # raw source の場合は y 列生成済み
        train_df[label_col] = task.make_labels(train_df.rename(columns={"close": "Close"}, errors="ignore"))
        test_df[label_col] = task.make_labels(test_df.rename(columns={"close": "Close"}, errors="ignore"))

    feature_cols = [c for c in train_df.columns if c not in exclude_cols and c not in ("Close", "close")]

    X_train = train_df[feature_cols].dropna()
    y_train = train_df.loc[X_train.index, label_col]
    X_test = test_df[feature_cols].dropna()

    # モデル学習
    mm = ModelManager()
    mm.create_model(args.model_type, model_name)
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
        market=args.market,
        symbol=args.symbol,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
    )

    # Close 列を test_df から準備
    close_col = "Close" if "Close" in test_df.columns else "close"
    result_df, metrics = backtester.simulate_trading(test_df.loc[X_test.index], signal)

    return result_df, metrics, None


def run_walk_forward(args):
    """Walk-Forward バックテスト"""
    from src.models.model_manager import ModelManager
    from src.strategy.signal_generator import SignalGenerator
    from src.backtest.walk_forward import WalkForwardValidator
    from src.backtest.task import ReturnRegressionTask

    task = build_task(args.task, args.threshold)
    model_name = args.model_name or f"Backtest{args.model_type}"

    mm = ModelManager()
    mm.create_model(args.model_type, model_name)

    wfv = WalkForwardValidator(
        market=args.market,
        symbol=args.symbol,
        model_manager=mm,
        signal_generator=SignalGenerator(),
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        n_splits=args.n_splits,
    )

    results_df = wfv.run(model_name=model_name, task=task)
    return None, None, results_df


def save_results(
    result_df,
    metrics,
    wf_df,
    market: str,
    symbol: str,
    task_name: str,
):
    """結果を CSV に保存する"""
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


def print_metrics(metrics: dict, label: str = ""):
    if not metrics:
        return
    print(f"\n{'='*50}")
    if label:
        print(f" {label}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print(f"{'='*50}")


def main():
    args = parse_args()
    print(f"\nバックテスト開始: {args.market}/{args.symbol} | task={args.task} | model={args.model_type}")

    if args.walk_forward:
        print(f"モード: Walk-Forward (n_splits={args.n_splits})")
        result_df, metrics, wf_df = run_walk_forward(args)
    else:
        print(f"モード: 単一期間 (train_ratio={args.train_ratio})")
        result_df, metrics, wf_df = run_single(args)
        print_metrics(metrics, label=f"{args.market}/{args.symbol} - {args.task}")

    save_results(result_df, metrics, wf_df, args.market, args.symbol, args.task)
    print("\n完了")


if __name__ == "__main__":
    main()
