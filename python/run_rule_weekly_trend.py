"""
週ごとのルールの採点の JSON の表を出すスクリプト（#1906。CLI ラッパー）

results/rule_accuracy/weekly の YYYY-Www.json を読み、次の2つの表を CSV か Markdown で出す。
(1) 週・市場・ルール・向き・ホライズンごとの n・hit_rate・baseline_rate・edge・mean_signed_return
(2) 週・市場ごとの n_outcomes と rule_best の出入り（added・removed・changed）の件数
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.rule_weekly_trend import (
    filter_weekly_outcomes,
    format_weekly_markdown,
    load_weekly_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="週ごとのルールの採点と rule_best の出入りを表にする"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="週の JSON の置き場（省略時は results/rule_accuracy/weekly）",
    )
    parser.add_argument("--market", type=str, default=None, help="この市場の JSON だけ読む")
    parser.add_argument("--rule", type=str, default=None, help="このルールの行だけ出す")
    parser.add_argument("--horizon", type=str, default=None, help="このホライズンの行だけ出す")
    parser.add_argument("--direction", type=str, default=None, help="この向きの行だけ出す")
    parser.add_argument(
        "--table",
        choices=["both", "outcomes", "diff"],
        default="both",
        help="出す表（outcomes は採点、diff は rule_best の出入り）",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    weekly_dir = args.dir or os.path.join(get_results_dir(), "rule_accuracy", "weekly")
    outcomes, diff = load_weekly_trend(weekly_dir, market=args.market)
    outcomes = filter_weekly_outcomes(
        outcomes, rule=args.rule, horizon=args.horizon, direction=args.direction
    )

    tables = []
    if args.table in ("both", "outcomes"):
        tables.append(outcomes)
    if args.table in ("both", "diff"):
        tables.append(diff)
    if args.format == "csv":
        text = "\n".join(t.to_csv(index=False) for t in tables)
    else:
        text = "\n\n".join(format_weekly_markdown(t) for t in tables) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"週ごとのルールの採点の表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
