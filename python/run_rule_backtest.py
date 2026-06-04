r"""
ルールベースバックテスト CLI

テクニカル指標ルールを使った売買戦略をバックテストし、
勝率・総利益でランキングを出力する。

使用例:
    # 単一銘柄・全ルール比較
    py run_rule_backtest.py --market jp --symbol 6920 --compare-all

    # 自動スクリーニング（高ボラ上位10銘柄 × 全ルール）
    py run_rule_backtest.py --market jp --screen-volatile --top-n 10

    # 特定ルールのみ
    py run_rule_backtest.py --market jp --symbol 9984 --rule volume_breakout

    # 期間・資金設定を変更
    py run_rule_backtest.py --market jp --symbol 6920 \\
        --start 2022-01-01 --end 2025-01-01 \\
        --initial-cash 500000 --stop-loss 0.05
"""

import argparse
import sys
from pathlib import Path

# Windowsコンソール(cp932)でUTF-8出力を可能にする
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("cp932", "shift-jis", "shift_jis"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.rule_backtester import print_rule_ranking  # noqa: E402
from src.backtest.rule_backtester import run_multi_symbol_rule_backtest  # noqa: E402
from src.backtest.rule_backtester import run_rule_backtest  # noqa: E402; noqa: E402
from src.backtest.rules import ALL_RULES  # noqa: E402
from src.backtest.rules.technical import BollingerBandRule  # noqa: E402
from src.backtest.rules.technical import EMAMomentumRule  # noqa: E402
from src.backtest.rules.technical import MACDRSIRule  # noqa: E402
from src.backtest.rules.technical import RSIContrarianRule  # noqa: E402
from src.backtest.rules.technical import VolatilityBreakoutRule  # noqa: E402
from src.backtest.rules.technical import VolumeBreakoutRule  # noqa: E402; noqa: E402
from src.backtest.screener import screen_volatile_symbols  # noqa: E402

RULE_MAP = {
    "volume_breakout": VolumeBreakoutRule(),
    "ema_momentum": EMAMomentumRule(),
    "rsi_contrarian": RSIContrarianRule(),
    "bollinger_band": BollingerBandRule(),
    "macd_rsi": MACDRSIRule(),
    "volatility_breakout": VolatilityBreakoutRule(),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="テクニカルルールベースバックテスト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 銘柄指定
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--symbol", help="銘柄コード（例: 6920, 9984）")
    target.add_argument(
        "--screen-volatile",
        action="store_true",
        help="高ボラ銘柄を自動スクリーニング",
    )

    p.add_argument("--market", default="jp", choices=["jp", "us"], help="マーケット (default: jp)")
    p.add_argument("--top-n", type=int, default=10, help="スクリーニング上位N銘柄 (default: 10)")

    # ルール選択
    rule_group = p.add_mutually_exclusive_group()
    rule_group.add_argument(
        "--compare-all",
        action="store_true",
        default=True,
        help="全ルールを比較（デフォルト）",
    )
    rule_group.add_argument(
        "--rule",
        choices=list(RULE_MAP.keys()),
        help="特定ルールのみ実行",
    )

    # 期間
    p.add_argument("--start", default="2022-01-01", help="バックテスト開始日 (default: 2022-01-01)")
    p.add_argument("--end", default="2025-01-01", help="バックテスト終了日 (default: 2025-01-01)")

    # スクリーニング期間（自動スクリーニング時のみ使用）
    p.add_argument(
        "--screen-start",
        default="2024-01-01",
        help="スクリーニング期間開始日 (default: 2024-01-01)",
    )
    p.add_argument(
        "--screen-end",
        default="2024-12-31",
        help="スクリーニング期間終了日 (default: 2024-12-31)",
    )

    # 取引条件
    p.add_argument(
        "--initial-cash", type=float, default=1_000_000, help="初期資金 (default: 1000000)"
    )
    p.add_argument("--fee-rate", type=float, default=0.001, help="片道手数料率 (default: 0.001)")
    p.add_argument("--slippage", type=float, default=0.001, help="スリッページ率 (default: 0.001)")
    p.add_argument(
        "--stop-loss",
        type=float,
        default=0.07,
        help="ストップロス率 (default: 0.07 = 7%%)",
    )
    p.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="テイクプロフィット率 (default: なし)",
    )
    p.add_argument(
        "--no-stop-loss",
        action="store_true",
        help="ストップロス無効化",
    )

    return p.parse_args()


def main() -> None:
    from src.orchestration.port_wiring import wire_ports

    wire_ports()
    args = parse_args()

    stop_loss = None if args.no_stop_loss else args.stop_loss
    rules = [RULE_MAP[args.rule]] if args.rule else ALL_RULES

    print(f"\n{'=' * 70}")
    print("  ルールベースバックテスト")
    print(f"  期間: {args.start} → {args.end}")
    print(f"  初期資金: JPY{args.initial_cash:,.0f}  手数料: {args.fee_rate:.3%}  SL: {stop_loss}")
    print(f"  ルール数: {len(rules)}")
    print(f"{'=' * 70}\n")

    if args.screen_volatile:
        print(f"高ボラ銘柄スクリーニング中 ({args.market})...")
        symbols = screen_volatile_symbols(
            market=args.market,
            top_n=args.top_n,
            screen_start=args.screen_start,
            screen_end=args.screen_end,
        )
        if not symbols:
            print("スクリーニング結果が空です。終了します。")
            sys.exit(1)

        print(f"\n選定銘柄: {symbols}\n")

        result_df = run_multi_symbol_rule_backtest(
            market=args.market,
            symbols=symbols,
            rules=rules,
            start=args.start,
            end=args.end,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            slippage=args.slippage,
            stop_loss_pct=stop_loss,
            take_profit_pct=args.take_profit,
        )

        if result_df.empty:
            print("バックテスト結果が空です。")
            sys.exit(1)

        # 銘柄別サマリー
        print("\n\n===== 銘柄 × ルール 総合ランキング =====")
        print(
            f"{'順位':<4} {'銘柄':>6} {'法則名':<24} {'勝率':>6} "
            f"{'総利益(円)':>12} {'取引数':>6} {'MDD':>8}"
        )
        print("-" * 70)
        for rank, (_, row) in enumerate(result_df.iterrows(), 1):
            profit_str = f"JPY{row['net_profit']:>+,.0f}"
            mdd_str = f"{row['max_drawdown']:.1%}"
            print(
                f"{rank:<4} {row['symbol']:>6} {row['rule']:<24} {row['win_rate']:>5.1%} "
                f"{profit_str:>12} {row['num_trades']:>6} {mdd_str:>8}"
            )

        # ルール別集計（平均勝率・合計利益）
        print("\n\n===== ルール別サマリー（全銘柄平均） =====")
        summary = (
            result_df.groupby("rule")
            .agg(
                avg_win_rate=("win_rate", "mean"),
                total_profit=("net_profit", "sum"),
                avg_trades=("num_trades", "mean"),
                count=("symbol", "count"),
            )
            .sort_values(["avg_win_rate", "total_profit"], ascending=[False, False])
        )
        print(
            f"{'法則名':<24} {'平均勝率':>8} {'合計利益(円)':>14} "
            f"{'平均取引数':>10} {'銘柄数':>6}"
        )
        print("-" * 65)
        for rule_name, row in summary.iterrows():
            profit_str = f"JPY{row['total_profit']:>+,.0f}"
            print(
                f"{rule_name:<24} {row['avg_win_rate']:>7.1%} "
                f"{profit_str:>14} {row['avg_trades']:>10.1f} {row['count']:>6.0f}"
            )

    else:
        results = run_rule_backtest(
            market=args.market,
            symbol=args.symbol,
            rules=rules,
            start=args.start,
            end=args.end,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            slippage=args.slippage,
            stop_loss_pct=stop_loss,
            take_profit_pct=args.take_profit,
        )

        if not results:
            print("バックテスト結果が空です。")
            sys.exit(1)

        print_rule_ranking(results, args.market, args.symbol, args.initial_cash)

        # 詳細表示
        best = results[0]
        print(f"\n最優秀ルール: [{best['rule']}] {best['description']}")
        print(f"  勝率      : {best['win_rate']:.1%}")
        print(f"  総利益    : JPY{best['net_profit']:+,.0f}")
        print(f"  総リターン: {best['total_return']:.2%}")
        print(f"  取引数    : {best['num_trades']} 回")
        print(f"  PF        : {best['profit_factor']:.2f}")
        print(f"  最大DD    : {best['max_drawdown']:.1%}")
        print(f"  平均利益  : {best['avg_win']:.2%} / 平均損失: {best['avg_loss']:.2%}")


if __name__ == "__main__":
    main()
