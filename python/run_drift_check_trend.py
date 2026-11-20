"""
日次ドリフトチェックの流れの表を出すスクリプト（#2073。CLI ラッパー）

results/drift_check の日ごとの JSON を読み、閾値超過の割合・再学習の数などを日ごとに1行の表にし、
CSV か Markdown で出す。読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.drift_check_trend import (
    format_drift_check_csv,
    format_drift_check_markdown,
    load_drift_check_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="日次ドリフトチェックの流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="ドリフトチェックの記録の置き場（省略時は results/drift_check）",
    )
    parser.add_argument("--tail", type=int, default=None, help="直近 N 日だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    record_dir = args.dir or os.path.join(get_results_dir(), "drift_check")
    df = load_drift_check_trend(record_dir)
    if args.tail is not None:
        df = df.tail(args.tail).reset_index(drop=True)

    text = (
        format_drift_check_csv(df)
        if args.format == "csv"
        else format_drift_check_markdown(df) + "\n"
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
        logger.critical(f"ドリフトチェックの流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
