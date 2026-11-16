"""
影の比べの記録の流れの表を出すスクリプト（#1863。CLI ラッパー）

results/accuracy/shadow の週ごとの JSON を読み、週ごとの challenger と production の
的中率とその差・勝ち負け・定数の予想・昇格を飛ばした理由と、勝たない週と定数の週の
続いた数の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.shadow_trend import format_shadow_markdown, load_shadow_trend
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="影の比べの記録の流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="影の比べの JSON の置き場（省略時は results/accuracy/shadow）",
    )
    parser.add_argument("--tail", type=int, default=None, help="直近 N 週だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    shadow_dir = args.dir or os.path.join(get_results_dir(), "accuracy", "shadow")
    df = load_shadow_trend(shadow_dir)
    if args.tail:
        df = df.tail(args.tail)

    text = df.to_csv(index=False) if args.format == "csv" else format_shadow_markdown(df) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"影の比べの記録の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
