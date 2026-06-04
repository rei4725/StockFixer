"""
yfinance 財務取得・保存スクリプト。

全銘柄: py run_fetch_fundamentals.py --market us
単一銘柄: py run_fetch_fundamentals.py --market us --symbol AAPL

対象銘柄は stock_features テーブルの ``get_all_symbols()`` から取得し、指定
マーケットで絞り込む。1銘柄ずつ取得して ``upsert_fundamentals`` で保存する。
取得失敗銘柄はスキップしてログに残し、全体は止めない。

⚠️ 取得データは最新スナップショットのみ・PIT 非対応。ライブスクリーン専用。
"""

import argparse
import sys

from src.market_data.fundamentals import fetch_fundamentals
from src.utils.db.stock_features import get_all_symbols
from src.utils.db.stock_fundamentals import upsert_fundamentals
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run(market: str, symbol: str = None) -> None:
    """指定マーケット（または単一銘柄）の財務を取得・保存する。"""
    if symbol:
        targets = [(market, symbol)]
    else:
        targets = [(m, s) for m, s in get_all_symbols() if m == market]

    if not targets:
        logger.warning(f"対象銘柄がありません: market={market}")
        return

    success = 0
    for mkt, sym in targets:
        record = fetch_fundamentals(mkt, sym)
        if record is None:
            logger.warning(f"[スキップ] 財務取得失敗: {mkt}/{sym}")
            continue
        upsert_fundamentals(record)
        success += 1

    logger.info(f"財務取得完了: {success}/{len(targets)} 銘柄 (market={market})")


def main() -> None:
    parser = argparse.ArgumentParser(description="yfinance 財務取得・保存スクリプト")
    parser.add_argument("--market", type=str, required=True, help="市場名（例: us, jp）")
    parser.add_argument("--symbol", type=str, default=None, help="銘柄コード（例: AAPL）")
    args = parser.parse_args()
    run(args.market, args.symbol)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"財務取得 異常終了: {e}", exc_info=True)
        sys.exit(1)
