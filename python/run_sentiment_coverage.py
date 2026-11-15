"""
センチメントの埋まり具合の診断（CLIラッパー）

stock_features から読み、市場ごとの行数・ニュースのある行・0点の割合・
最後に0でなかった日を表で出す。採点と特徴量は変えない。
"""

import argparse
import sys

from src.market_data.sentiment_coverage import load_sentiment_rows, sentiment_coverage
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="市場ごとのセンチメントの埋まり具合を表で出す")
    parser.add_argument(
        "--since", type=str, default=None, help="この日以後の行だけを数える（YYYY-MM-DD）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    table = sentiment_coverage(load_sentiment_rows(since=args.since))
    print(table.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"センチメントの診断 異常終了: {e}", exc_info=True)
        sys.exit(1)
