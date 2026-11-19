"""
業種の上限で落とした業種ごとの数の流れの表を出すスクリプト（#2043。CLI ラッパー）

results/orders（8:50）の日ごとの JSON の sector_cap_by_sector を読み、日・業種ごとの表か、
業種ごとの合計・出た日数の表を CSV か Markdown で出す。16:30 の orders/rule は読まない。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.order_sector_cap_trend import (
    build_sector_cap_daily,
    build_sector_cap_summary,
    format_sector_cap_csv,
    format_sector_cap_markdown,
    load_sector_cap_days,
    tail_days,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="業種の上限で落とした業種ごとの数の流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="8:50 の発注の記録の置き場（省略時は results/orders）",
    )
    parser.add_argument(
        "--table",
        choices=["daily", "summary"],
        default="daily",
        help="daily は日・業種ごと、summary は業種ごとの合計と出た日数",
    )
    parser.add_argument("--tail", type=int, default=None, help="鍵のある直近 N 日だけ使う")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    orders_dir = args.dir or os.path.join(get_results_dir(), "orders")
    days = tail_days(load_sector_cap_days(orders_dir), args.tail)
    df = build_sector_cap_summary(days) if args.table == "summary" else build_sector_cap_daily(days)

    text = (
        format_sector_cap_csv(df) if args.format == "csv" else format_sector_cap_markdown(df) + "\n"
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
        logger.critical(f"業種の上限の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
