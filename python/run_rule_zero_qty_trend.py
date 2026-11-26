"""
ルールの道で株数0で落ちた銘柄の流れの表を出すスクリプト（#2189。CLI ラッパー）

results/orders/rule（16:30）の日ごとの JSON の zero_qty_symbols を読み、日・銘柄ごとの表か、
銘柄ごとの落ちた日数・最後の理由の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.rule_zero_qty_trend import (
    build_zero_qty_daily,
    build_zero_qty_summary,
    format_zero_qty_csv,
    format_zero_qty_markdown,
    load_zero_qty_days,
    tail_days,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="ルールの道で株数0で落ちた銘柄の流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="16:30 のルールの道の記録の置き場（省略時は results/orders/rule）",
    )
    parser.add_argument(
        "--table",
        choices=["daily", "summary"],
        default="daily",
        help="daily は日・銘柄ごと、summary は銘柄ごとの落ちた日数と最後の理由",
    )
    parser.add_argument("--tail", type=int, default=None, help="鍵のある直近 N 日だけ使う")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    rule_dir = args.dir or os.path.join(get_results_dir(), "orders", "rule")
    days = tail_days(load_zero_qty_days(rule_dir), args.tail)
    df = build_zero_qty_summary(days) if args.table == "summary" else build_zero_qty_daily(days)

    text = format_zero_qty_csv(df) if args.format == "csv" else format_zero_qty_markdown(df) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"株数0の銘柄の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
