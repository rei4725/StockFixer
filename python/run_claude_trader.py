"""
run_claude_trader.py — Claude Opus トレード判断エージェント CLI

使い方:
    # ペーパートレード（デフォルト）
    py run_claude_trader.py

    # マーケット指定
    py run_claude_trader.py --market us

環境変数:
    CLAUDE_TRADER_ENABLED=true   # この設定がないと起動しない
    ANTHROPIC_API_KEY=sk-...     # Anthropic API キー（必須）
    CLAUDE_TRADER_MODEL          # モデル ID（デフォルト: claude-opus-4-7）
    CLAUDE_TRADER_THINKING_BUDGET # thinking budget tokens（デフォルト: 5000）
"""

import argparse
import sys

from src.domain.ports import TradeDiffSink
from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
from src.orchestration.port_wiring import wire_ports
from src.utils.logger import get_logger

logger = get_logger(__name__)

wire_ports()


def build_broker(mode: str, trade_diff_sink: TradeDiffSink):
    if mode == "live":
        import os

        from src.trading.brokers.kabu.kabu_client import KabuBroker

        api_password = os.environ.get("KABU_API_PASSWORD")
        if not api_password:
            logger.critical("live モードには KABU_API_PASSWORD が必要です")
            sys.exit(1)
        return KabuBroker(api_password=api_password)
    else:
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.trading.brokers.paper.paper_broker import PaperBroker

        return PaperBroker(
            market_data_port=YFinanceMarketDataAdapter(),
            trade_diff_sink=trade_diff_sink,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Claude Opus トレード判断エージェント")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="paper: ペーパートレード（デフォルト）, live: 本番",
    )
    parser.add_argument(
        "--market",
        default="jp",
        help="対象マーケット（デフォルト: jp）",
    )
    args = parser.parse_args(argv)

    from config.settings import CLAUDE_TRADER_ENABLED

    if not CLAUDE_TRADER_ENABLED:
        logger.error(
            "CLAUDE_TRADER_ENABLED=false のため起動しません。"
            "有効化するには環境変数 CLAUDE_TRADER_ENABLED=true を設定してください。"
        )
        return 1

    try:
        from src.trading.claude_agent import run_claude_trader

        trade_diff_sink = PostgresTradeDiffSink()
        broker = build_broker(args.mode, trade_diff_sink)
        stats = run_claude_trader(
            broker=broker,
            market=args.market,
            mode=args.mode,
            trade_diff_sink=trade_diff_sink,
        )
        print(
            f"完了 — 買い: {stats['buy_orders']} 売り: {stats['sell_orders']} "
            f"スキップ: {stats['skipped']} エラー: {stats['errors']}"
        )
        return 0
    except ImportError as e:
        logger.critical("依存パッケージが不足しています: %s", e)
        return 1
    except Exception as e:
        logger.critical("Claude トレーダー 致命的エラー: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
