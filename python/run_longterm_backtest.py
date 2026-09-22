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

from src.backtest.execution import DEFAULT_FEE_RATE
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.engine import run_longterm_backtest
from src.backtest.longterm.reporting import build_conclusion, save_results
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _non_negative_int(value: str) -> int:
    """execution_lag 用の argparse type。負数は明示的に拒否する。"""
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"execution_lag は 0 以上である必要があります: {value}")
    return ivalue


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
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="売買手数料率")
    parser.add_argument(
        "--slippage",
        type=float,
        default=None,
        help="片道スリッページ率（未指定なら市場別の既定値を使う）",
    )
    parser.add_argument("--benchmark", type=str, default="^GSPC", help="ベンチマークティッカー")
    parser.add_argument(
        "--execution-lag",
        type=_non_negative_int,
        default=1,
        help="エントリー約定をリスクリーン日から何営業日ずらすか（0=当日Close、負数は不可）",
    )
    return parser.parse_args()


def _print_summary(metrics: dict) -> None:
    print("\n===== 計測指標 =====")
    for key in (
        "initial_cash",
        "final_cash",
        "total_return",
        "cagr",
        "max_drawdown",
        "num_trades",
        "n_2x",
        "n_3x",
        "n_5x",
        "n_10x",
        "win_rate",
        "avg_win_multiple",
        "avg_loss",
        "avg_held_days",
        "sharpe_ratio",
        "profit_factor",
        "calmar_ratio",
        "benchmark_return",
        "alpha",
    ):
        print(f"  {key:>18}: {metrics.get(key)}")


def main():
    from src.orchestration.port_wiring import wire_ports

    wire_ports()
    args = parse_args()
    config = LongtermBacktestConfig.build(
        market=args.market,
        start=args.start,
        end=args.end,
        rescreen_freq=args.rescreen_freq,
        top_n=args.top_n,
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        benchmark_ticker=args.benchmark,
        execution_lag=args.execution_lag,
    )
    equity_df, metrics, trades_df = run_longterm_backtest(config)

    _print_summary(metrics)

    print("\n===== 結論 =====")
    print(build_conclusion(metrics, config))

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
