"""
統合モデルの学習の記録の流れの表を出すスクリプト（#2151。CLI ラッパー）

results/unified_training の JSON を読み、予想の上がる割合と実際の割合のずれ・木の数などを
1回の学習ごとに1行の表にし、CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.unified_training_trend import (
    format_unified_training_csv,
    format_unified_training_markdown,
    load_unified_training_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="統合モデルの学習の記録の流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="統合モデルの学習の記録の置き場（省略時は results/unified_training）",
    )
    parser.add_argument("--model", type=str, default=None, help="このモデル名の行だけ出す")
    parser.add_argument("--tail", type=int, default=None, help="（絞った後の）直近 N 行だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    record_dir = args.dir or os.path.join(get_results_dir(), "unified_training")
    df = load_unified_training_trend(record_dir, model_name=args.model)
    if args.tail is not None:
        df = df.tail(args.tail).reset_index(drop=True)

    text = (
        format_unified_training_csv(df)
        if args.format == "csv"
        else format_unified_training_markdown(df) + "\n"
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
        logger.critical(f"統合モデルの学習の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
