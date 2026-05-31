"""
運用監視ダッシュボード実行スクリプト（R-303 CLI ラッパー）

DuckDB と既存レポート結果から損益・ドリフト・paper/real 乖離・
実験結果を横断表示する read-only ダッシュボードを標準出力へ表示する。

使い方:
    py run_dashboard.py
    py run_dashboard.py --days 7
    py run_dashboard.py --drift-n 30
    py run_dashboard.py --days 7 --drift-n 30
"""

import argparse
import sys

from src.reporting.dashboard import run_dashboard
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StockFixer 運用監視ダッシュボードを表示する（read-only）"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="paper/real 乖離の集計期間（日）デフォルト: 30",
    )
    parser.add_argument(
        "--drift-n",
        type=int,
        default=20,
        help="ドリフト判定に使う直近サンプル数（銘柄ごと）デフォルト: 20",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dashboard(recent_days=args.days, drift_n=args.drift_n)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("ダッシュボード表示に失敗しました: %s", e, exc_info=True)
        sys.exit(1)
