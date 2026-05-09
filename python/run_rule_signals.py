"""
日次ルールシグナル CLI

DB に保存された最優秀ルールを使って本日のシグナルを生成し、
ペーパートレード注文を自動発注する。通常は平日の市場終了後（16:00）に実行。

使用例:
    # 全有効銘柄のシグナルを生成してペーパートレード発注
    py run_rule_signals.py --market jp

    # シグナルを表示するだけ（発注しない）
    py run_rule_signals.py --market jp --dry-run

    # Discord 通知なし
    py run_rule_signals.py --market jp --no-discord
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.prediction.rule_signal_pipeline import execute_rule_paper_trades, run_rule_signal_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="日次ルールシグナル生成 + ペーパートレード発注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--market", default="jp", choices=["jp", "us"])
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="シグナルを表示するのみ。注文は発行しない",
    )
    p.add_argument(
        "--no-discord",
        action="store_true",
        help="Discord 通知を送らない",
    )
    p.add_argument(
        "--budget-per-trade",
        type=float,
        default=100_000,
        help="1銘柄あたり最大投資額（円）（default: 100000）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    today = date.today()

    logger.info(f"=== 日次ルールシグナルジョブ開始 ({today}) ===")

    # 1. シグナル生成
    signals = run_rule_signal_pipeline(market=args.market, signal_date=today)

    if not signals:
        logger.warning("シグナルなし（週次評価 run_rule_evaluate.py を先に実行してください）")
        print("\nシグナルを生成できる有効ルールがありません。")
        print("先に `py run_rule_evaluate.py --market jp` を実行してください。")
        sys.exit(0)

    # 2. シグナル一覧を表示
    buy_signals = [s for s in signals if s["signal"] == 1]
    sell_signals = [s for s in signals if s["signal"] == -1]
    hold_signals = [s for s in signals if s["signal"] == 0]

    print(f"\n{'=' * 65}")
    print(f"  ルールシグナル  {today}  ({args.market})")
    print(f"{'=' * 65}")
    print(f"  BUY  : {len(buy_signals)} 銘柄")
    print(f"  SELL : {len(sell_signals)} 銘柄")
    print(f"  HOLD : {len(hold_signals)} 銘柄")
    print(f"{'=' * 65}")

    if buy_signals:
        print("\n[BUY シグナル]")
        print(f"{'銘柄':>6}  {'ルール':<22}  {'価格':>8}  {'勝率':>6}  {'純利益(円)':>12}")
        print("-" * 62)
        for s in buy_signals:
            price_str = f"{s['price']:>8,.0f}" if s.get("price") else "       ---"
            profit_str = f"JPY{s['net_profit']:>+,.0f}"
            print(
                f"{s['symbol']:>6}  {s['rule']:<22}  {price_str}  {s['win_rate']:>5.1%}  {profit_str:>12}"
            )

    if sell_signals:
        print("\n[SELL シグナル]")
        print(f"{'銘柄':>6}  {'ルール':<22}  {'価格':>8}  {'勝率':>6}")
        print("-" * 48)
        for s in sell_signals:
            price_str = f"{s['price']:>8,.0f}" if s.get("price") else "       ---"
            print(f"{s['symbol']:>6}  {s['rule']:<22}  {price_str}  {s['win_rate']:>5.1%}")

    # 3. ペーパートレード発注
    buy_orders = 0
    sell_orders = 0

    if not args.dry_run:
        print("\nペーパートレード発注中...")
        trade_stats = execute_rule_paper_trades(
            signals=signals,
            market=args.market,
            initial_budget_per_trade=args.budget_per_trade,
        )
        buy_orders = trade_stats["buy_orders"]
        sell_orders = trade_stats["sell_orders"]
        print(f"  BUY 発注: {buy_orders} 件 / SELL 発注: {sell_orders} 件")
    else:
        print("\n[dry-run] 発注はスキップしました")

    # 4. Discord 通知
    if not args.no_discord:
        try:
            from src.reporting.discord.discord_utils import send_rule_daily_signals

            send_rule_daily_signals(
                signals=signals,
                market=args.market,
                buy_orders=buy_orders,
                sell_orders=sell_orders,
            )
        except Exception as exc:
            logger.warning(f"Discord通知失敗: {exc}")

    logger.info(f"=== 日次ルールシグナルジョブ完了 ===")


if __name__ == "__main__":
    main()
