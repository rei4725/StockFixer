"""
日本株の銘柄マスタ（コード → 日本語社名）更新スクリプト。

py run_symbol_master.py

JPX の東証上場銘柄一覧を取得し、watchlist.json の jp 銘柄の社名と33業種を
symbol_master テーブルに保存する。一覧に無い銘柄はログに残してスキップする。
"""

import json
import sys

from src.market_data.symbol_master import build_jp_symbol_master
from src.utils.data_path_utils import get_watchlist_path
from src.utils.db.symbol_master import upsert_symbol_names
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    with open(get_watchlist_path(), encoding="utf-8") as f:
        symbols = json.load(f).get("jp", [])
    result = build_jp_symbol_master(symbols)
    saved = upsert_symbol_names(result.records)
    logger.info(f"銘柄マスタ更新完了: {saved}/{len(symbols)} 銘柄 (missing={result.missing})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"銘柄マスタ更新 異常終了: {e}", exc_info=True)
        sys.exit(1)
