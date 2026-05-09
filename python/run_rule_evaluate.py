"""
週次ルール評価 CLI

ウォッチリスト全銘柄をバックテストし、銘柄ごとに
「勝率50%以上 AND 総利益プラス」を満たす最優秀ルールを DuckDB に保存する。
通常は週次スケジューラー（日曜日）から呼び出される。

使用例:
    # 全銘柄を評価（約 30〜60 分かかる）
    py run_rule_evaluate.py --market jp

    # 特定銘柄のみ評価（テスト用）
    py run_rule_evaluate.py --market jp --symbols 6920 9984 7203

    # 評価済み結果を表示するだけ
    py run_rule_evaluate.py --market jp --show-only

    # バックテスト期間を変更
    py run_rule_evaluate.py --market jp --start 2021-01-01 --end 2025-01-01
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.rule_selector import evaluate_all_symbols, print_rule_summary
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="銘柄×ルール週次評価ジョブ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--market", default="jp", choices=["jp", "us"])
    p.add_argument("--symbols", nargs="+", help="評価対象銘柄（省略で全ウォッチリスト）")
    p.add_argument("--start", default="2022-01-01", help="バックテスト開始日")
    p.add_argument("--end", default="2025-01-01", help="バックテスト終了日")
    p.add_argument("--initial-cash", type=float, default=1_000_000)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--stop-loss", type=float, default=0.07)
    p.add_argument("--no-stop-loss", action="store_true")
    p.add_argument(
        "--show-only",
        action="store_true",
        help="バックテストを実行せず、保存済み結果を表示するだけ",
    )
    p.add_argument(
        "--no-discord",
        action="store_true",
        help="Discord 通知を送らない",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.show_only:
        print_rule_summary(args.market)
        return

    stop_loss = None if args.no_stop_loss else args.stop_loss

    logger.info("=== 週次ルール評価ジョブ開始 ===")
    logger.info(f"  市場: {args.market}  期間: {args.start} → {args.end}")

    summary = evaluate_all_symbols(
        market=args.market,
        backtest_start=args.start,
        backtest_end=args.end,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        stop_loss_pct=stop_loss,
        symbols=args.symbols,
    )

    print(f"\n{'=' * 50}")
    print(f"  ルール評価完了")
    print(f"  評価銘柄: {summary['evaluated']}")
    print(f"  有効ルール発見: {summary['effective']} 銘柄")
    print(f"  スキップ: {summary['skipped']} 銘柄")
    print(f"{'=' * 50}")

    print_rule_summary(args.market)

    if not args.no_discord:
        try:
            from src.reporting.discord.discord_utils import send_rule_evaluation_completion

            send_rule_evaluation_completion(
                evaluated=summary["evaluated"],
                effective=summary["effective"],
                skipped=summary["skipped"],
                market=args.market,
            )
        except Exception as exc:
            logger.warning(f"Discord通知失敗: {exc}")

    logger.info("=== 週次ルール評価ジョブ完了 ===")


if __name__ == "__main__":
    main()
