r"""
バックテスト実行スクリプト（ラッパー）

使用例:
    # 単一銘柄・単一期間
    py run_backtest.py --market jp --symbol 7203

    # Walk-Forwardに切り替え
    py run_backtest.py --market jp --symbol 7203 --walk-forward

    # モデル・閾値指定
    py run_backtest.py --market us --symbol AAPL \
        --model-type LightGBMModel --threshold 0.005

    # 生OHLCVからfeature再生成して実行
    py run_backtest.py --market jp --symbol 7203 --source raw

    # ストップロス・テイクプロフィット付き
    py run_backtest.py --market jp --symbol 7203 --stop-loss 0.05 --take-profit 0.10

    # アンサンブル予測（XGBoost+LightGBM平均）
    py run_backtest.py --market jp --symbol 7203 --ensemble

    # ポジションサイジング（予測確信度ベース）
    py run_backtest.py --market jp --symbol 7203 --position-sizing confidence
"""

import argparse
import os
import sys

from src.services.backtest_pipeline import (
    print_backtest_metrics,
    run_backtest_single,
    run_backtest_walk_forward,
    save_backtest_results,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
    parser.add_argument(
        "--model-name", type=str, default=None, help="モデル名 (default: Backtest{model_type})"
    )
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
        choices=["file", "api", "raw"],
        help="データソース: 'file'=DB特徴量, 'api'=yfinance直接取得, 'raw'=DBのOHLCVから再生成 (default: file)",
    )
    parser.add_argument("--walk-forward", action="store_true", help="Walk-Forward 検証を使用する")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-Forward の分割数 (default: 5)")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="単一期間モードの学習データ比率 (default: 0.8)",
    )
    parser.add_argument(
        "--initial-cash", type=float, default=1_000_000, help="初期資金 (default: 1,000,000)"
    )
    parser.add_argument("--fee-rate", type=float, default=0.001, help="取引手数料率 (default: 0.001)")
    parser.add_argument("--slippage", type=float, default=0.0, help="スリッページ (default: 0.0)")

    # リスク管理
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="ストップロス率 (例: 0.05=5%%下落で損切り)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="テイクプロフィット率 (例: 0.10=10%%上昇で利確)",
    )

    # ポジションサイジング
    parser.add_argument(
        "--position-sizing",
        type=str,
        default="full",
        choices=["full", "fixed", "confidence", "atr"],
        help="ポジションサイジング: full=全額, fixed=固定比率, confidence=予測確信度ベース, atr=ATR連動 (default: full)",
    )
    parser.add_argument(
        "--position-fraction",
        type=float,
        default=0.5,
        help="固定ポジション比率 (fixedモード用, default: 0.5)",
    )
    parser.add_argument(
        "--atr-risk-pct",
        type=float,
        default=0.02,
        help="ATRモード: 1トレードあたりのリスク割合 (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=1.0,
        help="ATRモード: ストップ幅とするATRの倍数 (default: 1.0)",
    )

    # アンサンブル
    parser.add_argument("--ensemble", action="store_true", help="XGBoost+LightGBMアンサンブル予測を使用する")

    # ベンチマーク比較
    parser.add_argument(
        "--benchmark",
        type=str,
        default="none",
        choices=["none", "n225", "sp500", "topix", "nasdaq"],
        help="比較するベンチマーク指数 (default: none)",
    )

    # グラフ出力
    parser.add_argument("--save-chart", action="store_true", help="バックテスト結果グラフ(PNG)をresults/に保存する")
    parser.add_argument(
        "--discord-chart",
        action="store_true",
        help="バックテスト結果グラフをDiscord Webhookに送信する（--save-chart も自動有効化）",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    model_label = "Ensemble(XGB+LGB)" if args.ensemble else args.model_type
    logger.info(f"バックテスト開始: {args.market}/{args.symbol} | task={args.task} | model={model_label}")

    if args.stop_loss:
        logger.info(f"  ストップロス: {args.stop_loss:.1%}")
    if args.take_profit:
        logger.info(f"  テイクプロフィット: {args.take_profit:.1%}")
    if args.position_sizing != "full":
        logger.info(f"  ポジションサイジング: {args.position_sizing}")

    common_kwargs = dict(
        market=args.market,
        symbol=args.symbol,
        model_type=args.model_type,
        model_name=args.model_name,
        task_name=args.task,
        threshold=args.threshold,
        source=args.source,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        position_sizing=args.position_sizing,
        position_fraction=args.position_fraction,
        atr_risk_pct=args.atr_risk_pct,
        atr_multiplier=args.atr_multiplier,
        ensemble=args.ensemble,
    )

    if args.walk_forward:
        logger.info(f"モード: Walk-Forward (n_splits={args.n_splits})")
        result_df, metrics, wf_df = run_backtest_walk_forward(
            **common_kwargs,
            n_splits=args.n_splits,
        )
    else:
        logger.info(f"モード: 単一期間 (train_ratio={args.train_ratio})")
        result_df, metrics, wf_df = run_backtest_single(
            **common_kwargs,
            start_date=args.start_date,
            end_date=args.end_date,
            train_ratio=args.train_ratio,
        )
        # ベンチマーク比較
        benchmark_info = None
        if args.benchmark != "none" and metrics and result_df is not None and not result_df.empty:
            from src.backtest.metrics import BENCHMARK_TICKERS, fetch_benchmark_returns

            bm_ticker = BENCHMARK_TICKERS.get(args.benchmark, args.benchmark)
            bm_start = (
                str(result_df["date"].min())[:10]
                if "date" in result_df.columns
                else args.start_date
            )
            bm_end = (
                str(result_df["date"].max())[:10] if "date" in result_df.columns else args.end_date
            )
            if bm_start and bm_end:
                benchmark_info = fetch_benchmark_returns(bm_ticker, bm_start, bm_end)

        print_backtest_metrics(
            metrics,
            label=f"{args.market}/{args.symbol} - {args.task}",
            benchmark=benchmark_info,
        )

    save_backtest_results(result_df, metrics, wf_df, args.market, args.symbol, args.task)

    # グラフ出力（--save-chart または --discord-chart 指定時）
    if (args.save_chart or args.discord_chart) and result_df is not None and not result_df.empty:
        from src.backtest.metrics import plot_backtest
        from src.utils.data_path_utils import get_results_dir

        out_dir = os.path.join(get_results_dir(), "backtest", f"{args.market}_{args.symbol}")
        chart_path = plot_backtest(
            result_df,
            metrics or {},
            out_dir,
            market=args.market,
            symbol=args.symbol,
            initial_cash=args.initial_cash,
        )
        if chart_path:
            logger.info(f"グラフ保存: {chart_path}")
            if args.discord_chart:
                from src.api.discord_utils import send_webhook_file

                send_webhook_file(
                    chart_path,
                    title=f"{args.market.upper()}/{args.symbol} バックテスト結果",
                )

    logger.info("完了")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"バックテスト 異常終了: {e}", exc_info=True)
        sys.exit(1)
