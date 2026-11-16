"""
採点の JSON の内訳の流れの表を出すスクリプト（#1927。CLI ラッパー）

results/accuracy の日ごとの JSON の by_sector か by_magnitude を読み、区分ごとの直近 N 日の
n_scored で重みを付けた平均（--view summary）か日ごとの行（--view daily）を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.accuracy_breakdown_trend import (
    BREAKDOWN_KINDS,
    load_breakdown_trend,
    summarize_breakdown_trend,
)
from src.reporting.accuracy_trend import format_trend_markdown
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="採点の JSON の業種・予想の大きさの内訳を表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="採点の JSON の置き場（省略時は results/accuracy）",
    )
    parser.add_argument("--kind", choices=sorted(BREAKDOWN_KINDS), default="sector")
    parser.add_argument("--window", type=int, default=5, help="重み付きの平均の日数（既定 5）")
    parser.add_argument("--model", type=str, default=None, help="このモデルの行だけ出す")
    parser.add_argument(
        "--view",
        choices=["summary", "daily"],
        default="summary",
        help="summary は区分ごとの重み付きの平均、daily は日ごとの行",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    accuracy_dir = args.dir or os.path.join(get_results_dir(), "accuracy")
    df = load_breakdown_trend(accuracy_dir, kind=args.kind)
    if args.model:
        df = df[df["model"] == args.model]
    if args.view == "summary":
        df = summarize_breakdown_trend(df, window=args.window)

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
        logger.critical(f"採点の JSON の内訳の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
