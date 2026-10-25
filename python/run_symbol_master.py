"""
日本株の銘柄マスタ（コード → 日本語社名）更新スクリプト。

py run_symbol_master.py

JPX の東証上場銘柄一覧を取得し、watchlist.json の jp 銘柄の社名と33業種を
symbol_master テーブルに保存する。一覧に無い銘柄はログに残してスキップする。
"""

import sys

from src.market_data.symbol_master import refresh_jp_symbol_master
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    refresh_jp_symbol_master()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"銘柄マスタ更新 異常終了: {e}", exc_info=True)
        sys.exit(1)
