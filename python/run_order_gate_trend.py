"""
発注の回の記録の流れの表を出すスクリプト（#1836。CLI ラッパー）

results/orders（8:50）と results/orders/rule（16:30）の日ごとの JSON を読み、日・道ごとの
発注の数・飛ばした理由の数・止まりの印・保有の判定漏れの数の表を CSV か Markdown で出す。
読むだけで、DB にもファイルにも書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import os
import sys

from src.reporting.order_gate_trend import (
    format_order_gate_csv,
    format_order_gate_markdown,
    load_order_gate_trend,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="発注の回の記録の流れを表にする")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="発注の記録の置き場（省略時は results/orders。16:30 の記録はその下の rule）",
    )
    parser.add_argument(
        "--route", choices=["0850", "1630"], default=None, help="この道の行だけ出す"
    )
    parser.add_argument("--tail", type=int, default=None, help="道ごとの直近 N 日だけ出す")
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    orders_dir = args.dir or os.path.join(get_results_dir(), "orders")
    df = load_order_gate_trend(orders_dir)
    if args.route:
        df = df[df["route"] == args.route]
    if args.tail:
        df = df.groupby("route", sort=False).tail(args.tail)

    text = (
        format_order_gate_csv(df) if args.format == "csv" else format_order_gate_markdown(df) + "\n"
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
        logger.critical(f"発注の回の記録の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
