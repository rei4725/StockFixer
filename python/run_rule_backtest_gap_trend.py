"""
ルールの採点の JSON の backtest_gap の流れの表を出すスクリプト（#1891。CLI ラッパー）

results/rule_accuracy の日ごとの JSON を読み、日・市場・ルール・向き（buy・sell・combined）ごとの
バックテストの勝率と錨の5日の的中率とその差などの表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.rule_backtest_gap_trend import (
    filter_gap_trend,
    format_gap_trend_markdown,
    load_rule_backtest_gap_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ルールの採点の JSON のバックテストの勝率と5日の的中率の差の流れを表にする"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="ルールの採点の JSON の置き場（省略時は results/rule_accuracy）",
    )
    parser.add_argument("--market", type=str, default=None, help="この市場の JSON だけ読む")
    parser.add_argument("--rule", type=str, default=None, help="このルールの行だけ出す")
    parser.add_argument(
        "--direction", type=str, default=None, help="この向きの行だけ出す（buy・sell・combined）"
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    accuracy_dir = args.dir or os.path.join(get_results_dir(), "rule_accuracy")
    df = load_rule_backtest_gap_trend(accuracy_dir, market=args.market)
    df = filter_gap_trend(df, rule=args.rule, direction=args.direction)

    text = df.to_csv(index=False) if args.format == "csv" else format_gap_trend_markdown(df) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"backtest_gap の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
