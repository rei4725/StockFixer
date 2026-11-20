"""
昇格ゲートの流れの表を出すスクリプト（#2073。CLI ラッパー）

results/promotion の昇格ゲートの JSON を読み、回ごとに基準ごとの合否・値・閾値と、続けて落ちた回数の
表を CSV か Markdown で出す。読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.promotion_gate_trend import (
    format_promotion_gate_csv,
    format_promotion_gate_markdown,
    load_promotion_gate_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="昇格ゲートの流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="昇格ゲートの記録の置き場（省略時は results/promotion）",
    )
    parser.add_argument("--tail", type=int, default=None, help="直近 N 回だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.dir or os.path.join(get_results_dir(), "promotion")
    # 続けて落ちた回数は全部の回で数えてから、直近 N 回を切り出す
    df = load_promotion_gate_trend(results_dir)
    if args.tail is not None:
        df = df.tail(args.tail).reset_index(drop=True)

    text = (
        format_promotion_gate_csv(df)
        if args.format == "csv"
        else format_promotion_gate_markdown(df) + "\n"
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"昇格ゲートの流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
