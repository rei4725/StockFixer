"""
月次レポートの流れの表を出すスクリプト（#2025。CLI ラッパー）

results/monthly の YYYY-MM_report.md を読み、月ごとに1行の KPI・週次 Hit Rate・
paper/real 乖離・ドリフト件数の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.monthly_report_trend import (
    format_monthly_trend_markdown,
    load_monthly_report_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="月次レポートの流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="月次レポートの置き場（省略時は results/monthly）",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    monthly_dir = args.dir or os.path.join(get_results_dir(), "monthly")
    df = load_monthly_report_trend(monthly_dir)

    text = (
        df.to_csv(index=False) if args.format == "csv" else format_monthly_trend_markdown(df) + "\n"
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
        logger.critical(f"月次レポートの流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
