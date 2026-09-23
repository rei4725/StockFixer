"""合成ルートが本物のアダプタを注入していることの検証。

Phase 4a で生まれた失敗様式への対処。DI 後は「呼び出しがあれば書かれた」が
成り立たず、合成ルートが InMemory 実装を本番側へ貼り間違えても
全テストが緑のまま本番の書き込みだけが止まる。合成ルートそのものを検証する。
"""

import unittest
from unittest.mock import MagicMock, patch

from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink


class TestRunAutoTradeCompositionRoot(unittest.TestCase):
    def test_paper_broker_gets_the_injected_sink(self):
        import run_auto_trade

        sink = PostgresTradeDiffSink()
        broker = run_auto_trade.build_broker("paper", sink)

        self.assertIs(broker._trade_diff_sink, sink)

    def test_main_injects_postgres_adapters(self):
        import run_auto_trade

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        with patch("src.trading.execution.run_daily_orders", _capture), patch.object(
            run_auto_trade, "build_broker", return_value=MagicMock()
        ):
            rc = run_auto_trade.main(["--mode", "paper", "--market", "jp"])

        self.assertEqual(rc, 0)
        self.assertIsInstance(captured["order_run_sink"], PostgresOrderRunSink)
        self.assertIsInstance(captured["trade_diff_sink"], PostgresTradeDiffSink)

    def test_broker_and_pipeline_share_one_sink(self):
        """broker に渡した Sink と run_daily_orders に渡す Sink が同一実体であること。"""
        import run_auto_trade

        captured: dict = {}
        seen_sinks: list = []

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        def _build(mode, trade_diff_sink):
            seen_sinks.append(trade_diff_sink)
            return MagicMock()

        with patch("src.trading.execution.run_daily_orders", _capture), patch.object(
            run_auto_trade, "build_broker", _build
        ):
            run_auto_trade.main(["--mode", "paper"])

        self.assertIs(seen_sinks[0], captured["trade_diff_sink"])


class TestDailyJobCompositionRoot(unittest.TestCase):
    def test_run_daily_auto_order_injects_postgres_adapters(self):
        from src.orchestration.jobs import daily

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {
                "buy_orders": 0,
                "sell_orders": 0,
                "short_orders": 0,
                "skipped": 0,
                "skipped_min_change": 0,
                "errors": 0,
                "trading_stopped": False,
                "total_turnover": 0.0,
            }

        with patch("src.trading.execution.run_daily_orders", _capture), patch(
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"
        ):
            try:
                daily.run_daily_auto_order()
            except Exception:
                pass  # Discord 通知など後続処理の失敗はここでは問わない

        self.assertIsInstance(captured.get("order_run_sink"), PostgresOrderRunSink)
        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)


if __name__ == "__main__":
    unittest.main()
