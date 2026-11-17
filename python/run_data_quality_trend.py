"""
品質の記録 data_quality_log の流れの表を出すスクリプト（#1951。CLI ラッパー）

data_quality_log を checked_at の範囲で読み、日・市場・check_name・level ごとに件数と銘柄の数の
表を CSV か Markdown で出す。読むだけで、DB にもファイルにも書かない（--output を渡したときだけ
そのファイルに書く）。
"""

import argparse
import sys
from datetime import datetime

from src.reporting.data_quality_trend import (
    build_data_quality_trend,
    format_data_quality_trend_markdown,
)
from src.utils.db.quality_log import load_quality_logs
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="品質の警告の日・市場・check_name・level ごとの件数と銘柄の数の流れを表にする"
    )
    parser.add_argument("--market", type=str, default=None, help="この市場の行だけ読む")
    parser.add_argument(
        "--since",
        type=datetime.fromisoformat,
        default=None,
        help="checked_at がこの日時（YYYY-MM-DD。tz が無ければ UTC）以後の行だけ読む",
    )
    parser.add_argument(
        "--until",
        type=datetime.fromisoformat,
        default=None,
        help="checked_at がこの日時（YYYY-MM-DD。tz が無ければ UTC）より前の行だけ読む",
    )
    parser.add_argument("--format", choices=["csv", "markdown"], default="markdown")
    parser.add_argument("--output", type=str, default=None, help="書き出す先（省略時は標準出力）")
    return parser.parse_args()


def main():
    args = parse_args()
    df = build_data_quality_trend(
        load_quality_logs(since=args.since, until=args.until, market=args.market)
    )

    text = (
        df.to_csv(index=False)
        if args.format == "csv"
        else format_data_quality_trend_markdown(df) + "\n"
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
        logger.critical(f"data_quality_log の流れの表の作成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
