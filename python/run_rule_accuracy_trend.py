"""
ルールの採点の JSON の流れの表を出すスクリプト（#1837。CLI ラッパー）

results/rule_accuracy の日ごとの JSON を読み、日・市場・窓・ルール・向き・ホライズンごとの
n と hit_rate と baseline_rate と edge と mean_signed_return の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.rule_accuracy_trend import (
    filter_rule_trend,
    format_rule_trend_markdown,
    load_rule_accuracy_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="ルールの採点の JSON の edge の流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="ルールの採点の JSON の置き場（省略時は results/rule_accuracy）",
    )
    parser.add_argument("--market", type=str, default=None, help="この市場の JSON だけ読む")
    parser.add_argument("--rule", type=str, default=None, help="このルールの行だけ出す")
    parser.add_argument("--window", type=str, default=None, help="この窓の行だけ出す（28d・all）")
    parser.add_argument("--horizon", type=str, default=None, help="このホライズンの行だけ出す")
    parser.add_argument("--direction", type=str, default=None, help="この向きの行だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    accuracy_dir = args.dir or os.path.join(get_results_dir(), "rule_accuracy")
    df = load_rule_accuracy_trend(accuracy_dir, market=args.market)
    df = filter_rule_trend(
        df,
        rule=args.rule,
        window=args.window,
        horizon=args.horizon,
        direction=args.direction,
    )

    text = df.to_csv(index=False) if args.format == "csv" else format_rule_trend_markdown(df) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"ルールの採点の JSON の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
