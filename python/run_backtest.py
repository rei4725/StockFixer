"""
バックテスト実行スクリプト（ラッパー）

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

from src.services.backtest_pipeline import (
    run_backtest_single,
    run_backtest_walk_forward,
    save_backtest_results,
    print_backtest_metrics,
)


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


def main():
    args = parse_args()
    print(f"\nバックテスト開始: {args.market}/{args.symbol} | task={args.task} | model={args.model_type}")

    if args.walk_forward:
        print(f"モード: Walk-Forward (n_splits={args.n_splits})")
        result_df, metrics, wf_df = run_backtest_walk_forward(
            market=args.market,
            symbol=args.symbol,
            model_type=args.model_type,
            model_name=args.model_name,
            task_name=args.task,
            threshold=args.threshold,
            n_splits=args.n_splits,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            slippage=args.slippage,
        )
    else:
        print(f"モード: 単一期間 (train_ratio={args.train_ratio})")
        result_df, metrics, wf_df = run_backtest_single(
            market=args.market,
            symbol=args.symbol,
            model_type=args.model_type,
            model_name=args.model_name,
            task_name=args.task,
            threshold=args.threshold,
            source=args.source,
            start_date=args.start_date,
            end_date=args.end_date,
            train_ratio=args.train_ratio,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            slippage=args.slippage,
        )
        print_backtest_metrics(metrics, label=f"{args.market}/{args.symbol} - {args.task}")

    save_backtest_results(result_df, metrics, wf_df, args.market, args.symbol, args.task)
    print("\n完了")


if __name__ == "__main__":
    main()
