"""
8:50 の発注の要約 order_run_summary の流れの表を出すスクリプト（#1981。CLI ラッパー）

order_run_summary を run_at の範囲で読み、日・市場ごとに回の数・買い・売り・ショート・スキップの
件数と total_turnover の合計の表を CSV か Markdown で出す。読むだけで、DB にもファイルにも
書かない（--output を渡したときだけそのファイルに書く）。
"""

import argparse
import sys
from datetime import datetime

from src.infrastructure.persistence.order_run_repository import load_order_run_summaries
from src.reporting.order_run_trend import (
    build_order_run_trend,
    format_order_run_csv,
    format_order_run_markdown,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="8:50 の発注の日・市場ごとの回の数・件数・売買の金額の流れを表にする"
    )
    parser.add_argument("--market", type=str, default=None, help="この市場の行だけ読む")
    parser.add_argument(
        "--since",
        type=datetime.fromisoformat,
        default=None,
        help="run_at がこの日時（YYYY-MM-DD）以後の行だけ読む",
    )
    parser.add_argument(
        "--until",
        type=datetime.fromisoformat,
        default=None,
        help="run_at がこの日時（YYYY-MM-DD）より前の行だけ読む",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    df = build_order_run_trend(
        load_order_run_summaries(since=args.since, until=args.until, market=args.market)
    )

    text = (
        format_order_run_csv(df) if args.format == "csv" else format_order_run_markdown(df) + "\n"
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
        logger.critical(f"order_run_summary の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
