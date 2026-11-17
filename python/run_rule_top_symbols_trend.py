"""
ルールの採点の JSON の銘柄×ルールの流れの表を出すスクリプト（#1972。CLI ラッパー）

results/rule_accuracy の日ごとの JSON の top_symbols を読み、市場・銘柄・ルール・向きごとに
出た日数・初めと最後の日・最後の日の n_h5・hit_rate_h5・edge_h5・edge_h5 の最小と最大の表を
CSV か Markdown で出す。読むだけで、DB にもファイルにも書かない（--output を渡したときだけ
そのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.accuracy_trend import format_trend_markdown
from src.reporting.rule_top_symbols_trend import load_rule_top_symbols_trend
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ルールの採点の JSON の銘柄×ルールの組を日をまたいで表にする"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="ルールの採点の JSON の置き場（省略時は results/rule_accuracy）",
    )
    parser.add_argument("--min-days", type=int, default=1, help="出た日数がこれ以上の行だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    accuracy_dir = args.dir or os.path.join(get_results_dir(), "rule_accuracy")
    df = load_rule_top_symbols_trend(accuracy_dir)
    df = df[df["n_days"] >= args.min_days]

    text = df.to_csv(index=False) if args.format == "csv" else format_trend_markdown(df) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"銘柄×ルールの流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
