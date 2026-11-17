"""
採点の JSON の大外れの銘柄の流れの表を出すスクリプト（#1928。CLI ラッパー）

results/accuracy の日ごとの JSON の top_misses を読み、モデル・ホライズン・市場・銘柄ごとに
出た日数・初めと最後の日・abs_error の平均と最大・外れの向きの数の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.accuracy_trend import format_trend_markdown
from src.reporting.top_misses_trend import load_top_misses_trend
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="採点の JSON の大外れの銘柄を日をまたいで表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="採点の JSON の置き場（省略時は results/accuracy）",
    )
    parser.add_argument("--model", type=str, default=None, help="このモデルの行だけ出す")
    parser.add_argument("--min-days", type=int, default=1, help="出た日数がこれ以上の行だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    accuracy_dir = args.dir or os.path.join(get_results_dir(), "accuracy")
    df = load_top_misses_trend(accuracy_dir)
    if args.model:
        df = df[df["model"] == args.model]
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
        logger.critical(f"大外れの銘柄の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
