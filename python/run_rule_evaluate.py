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

    # バックテスト期間を変更（省略時は週次ジョブと同じ実行日までの直近 RULE_EVAL_YEARS 年）
    py run_rule_evaluate.py --market jp --start 2021-01-01 --end 2025-01-01

評価の失敗（summary の errors）が1件以上なら、知らせを送った後に終了コード 1 で終える。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.rule_selector import evaluate_all_symbols  # noqa: E402
from src.backtest.rule_selector import print_rule_summary  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="銘柄×ルール週次評価ジョブ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--market", default="jp", choices=["jp", "us"])
    p.add_argument("--symbols", nargs="+", help="評価対象銘柄（省略で全ウォッチリスト）")
    p.add_argument("--start", default=None, help="バックテスト開始日（省略で週次ジョブと同じ）")
    p.add_argument("--end", default=None, help="バックテスト終了日（省略で週次ジョブと同じ）")
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from config.execution import RULE_EVAL_END, RULE_EVAL_START, RULE_EVAL_YEARS
    from src.orchestration.jobs.rule_evaluation import rule_evaluation_window
    from src.orchestration.port_wiring import wire_ports
    from src.utils.japan_time import now_jst

    wire_ports()
    args = parse_args(argv)

    if args.show_only:
        print_rule_summary(args.market)
        return 0

    stop_loss = None if args.no_stop_loss else args.stop_loss
    # 週次ジョブと同じ rule_best に書くため、省いたときは同じ期間で評価する
    backtest_start, backtest_end = rule_evaluation_window(
        now_jst().date(),
        RULE_EVAL_YEARS,
        args.start or RULE_EVAL_START,
        args.end or RULE_EVAL_END,
    )

    logger.info("=== 週次ルール評価ジョブ開始 ===")
    logger.info(f"  市場: {args.market}  期間: {backtest_start} → {backtest_end}")

    summary = evaluate_all_symbols(
        market=args.market,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        stop_loss_pct=stop_loss,
        symbols=args.symbols,
    )
    errors = summary.get("errors", 0)

    print(f"\n{'=' * 50}")
    print("  ルール評価完了")
    print(f"  評価銘柄: {summary['evaluated']}")
    print(f"  有効ルール発見: {summary['effective']} 銘柄")
    print(f"  スキップ: {summary['skipped']} 銘柄")
    print(f"  評価の失敗: {errors} 件")
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
                errors=errors,
            )
        except Exception as exc:
            logger.warning(f"Discord通知失敗: {exc}")

    # 失敗した銘柄は rule_best が前の週の行のまま 16:30 のルールの発注に使われる（#1459）
    if errors > 0:
        logger.error("ルール評価で評価の失敗が %d 件あった", errors)
        return 1

    logger.info("=== 週次ルール評価ジョブ完了 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
