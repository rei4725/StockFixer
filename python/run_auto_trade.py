"""
run_auto_trade.py — 自動発注 CLIエントリーポイント

使い方:
    # ペーパートレード（デフォルト）
    py run_auto_trade.py

    # 本番接続（kabu STATION® API）
    py run_auto_trade.py --mode live

    # ペーパートレードの約定処理（翌朝に実行）
    py run_auto_trade.py --settle

オプション:
    --mode    paper (default) | live
    --market  jp (default)
    --settle  pending 注文を当日始値で約定処理する
"""

import argparse
import sys

from src.orchestration.port_wiring import wire_ports
from src.utils.logger import get_logger

logger = get_logger(__name__)

wire_ports()


def _build_broker(mode: str):
    """mode に応じた Broker インスタンスを返す"""
    if mode == "live":
        import os

        from src.trading.brokers.kabu.kabu_client import KabuBroker

        api_password = os.environ.get("KABU_API_PASSWORD")
        if not api_password:
            logger.critical(
                "live モードには環境変数 KABU_API_PASSWORD が必要です。"
                "kabu STATION® アプリを起動した上で設定してください。"
            )
            sys.exit(1)
        return KabuBroker(api_password=api_password)
    else:
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.prediction.db import upsert_paper_real_diff
        from src.trading.brokers.paper.paper_broker import PaperBroker

        return PaperBroker(
            market_data_port=YFinanceMarketDataAdapter(),
            record_diff=upsert_paper_real_diff,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockFixer 自動発注スクリプト")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="paper: ペーパートレード（デフォルト）, live: 本番（kabu STATION® API）",
    )
    parser.add_argument(
        "--market",
        default="jp",
        help="対象マーケット（デフォルト: jp）",
    )
    parser.add_argument(
        "--settle",
        action="store_true",
        help="ペーパートレードの pending 注文を約定処理する",
    )
    args = parser.parse_args()

    try:
        broker = _build_broker(args.mode)

        if args.settle:
            if args.mode != "paper":
                logger.warning("--settle は paper モードでのみ有効です")
                sys.exit(1)

            settled = broker.settle_pending_orders()
            print(f"約定処理完了: {len(settled)} 件")
            for s in settled:
                print(f"  {s['symbol']} {s['qty']}株 @ {s['fill_price']:.1f}円")
        else:
            from src.trading.execution import run_daily_orders

            stats = run_daily_orders(broker=broker, market=args.market, mode=args.mode)
            print(
                f"発注完了 — 買い: {stats['buy_orders']} 売り: {stats['sell_orders']} "
                f"スキップ: {stats['skipped']} エラー: {stats['errors']}"
            )

    except Exception as e:
        logger.critical(f"自動発注スクリプト 致命的エラー: {e}", exc_info=True)
        sys.exit(1)
