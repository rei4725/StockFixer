r"""長期コホート・バイ&ホールド・バックテスト実行スクリプト（Stage 1）

過去の各時点でトレンド・スクリーン(#429)を回し、上位候補にエントリー→保有/撤退
エンジン(#430)で数年ホールドする規律を実データでシミュレートし、ポートフォリオ
リターン・最大DD・到達倍率分布（何倍株を何件捕まえたか）を出力する。

⚠️ 生存者バイアス: 対象ユニバースは現存銘柄のみのため、結果は楽観方向に歪む。

使用例:
    py run_longterm_backtest.py --market us --start 2021-01-01 --end 2026-01-01 \
        --top-n 30 --max-positions 10
"""

import argparse
import sys

from src.backtest.longterm_backtest import (
    build_conclusion,
    run_longterm_backtest,
    save_results,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="長期コホート・バイ&ホールド・バックテストを実行する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--market", type=str, default="us", help="マーケット識別子")
    parser.add_argument("--start", type=str, default="2021-01-01", help="開始日 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2026-01-01", help="終了日 YYYY-MM-DD")
    parser.add_argument(
        "--rescreen-freq",
        type=str,
        default="quarterly",
        choices=["weekly", "monthly", "quarterly", "yearly"],
        help="リスクリーン頻度",
    )
    parser.add_argument("--top-n", type=int, default=30, help="スクリーンで返す候補数")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0, help="初期資金")
    parser.add_argument("--max-positions", type=int, default=10, help="同時保有上限")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="売買手数料率")
    parser.add_argument("--benchmark", type=str, default="^GSPC", help="ベンチマークティッカー")
    return parser.parse_args()


def _print_summary(metrics: dict) -> None:
    print("\n===== 計測指標 =====")
    for key in (
        "initial_cash",
        "final_cash",
        "total_return",
        "cagr",
        "max_drawdown",
        "n_trades",
        "n_2x",
        "n_3x",
        "n_5x",
        "n_10x",
        "win_rate",
        "avg_win_multiple",
        "avg_loss",
        "avg_held_days",
        "benchmark_return",
        "alpha",
    ):
        print(f"  {key:>18}: {metrics.get(key)}")


def main():
    args = parse_args()
    equity_df, metrics, trades_df = run_longterm_backtest(
        market=args.market,
        start=args.start,
        end=args.end,
        rescreen_freq=args.rescreen_freq,
        top_n=args.top_n,
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        fee_rate=args.fee_rate,
        benchmark_ticker=args.benchmark,
    )

    _print_summary(metrics)

    print("\n===== 結論 =====")
    print(build_conclusion(metrics, args.market, args.start, args.end, args.max_positions))

    if not equity_df.empty:
        equity_path, trades_path = save_results(equity_df, trades_df, args.market)
        print(f"\nCSV 保存: {equity_path}")
        print(f"CSV 保存: {trades_path}")
    else:
        print("\n対象データがないため CSV 保存をスキップしました。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"長期バックテスト 異常終了: {e}", exc_info=True)
        sys.exit(1)
