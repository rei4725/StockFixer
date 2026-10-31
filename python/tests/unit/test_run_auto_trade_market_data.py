"""手で流す発注の CLI が日次の道と同じ値段の取り元を渡し、注文の失敗で0以外を返すことの検証（#1450）。"""

import unittest
from unittest.mock import MagicMock, patch


class TestRunAutoTradeMarketData(unittest.TestCase):
    def _run(self, argv: list[str], errors: int = 0):
        import run_auto_trade

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": errors}

        with patch("src.trading.execution.run_daily_orders", _capture), patch.object(
            run_auto_trade, "build_broker", return_value=MagicMock()
        ), patch.object(run_auto_trade, "PostgresTradeDiffSink"), patch.object(
            run_auto_trade, "PostgresOrderRunSink"
        ), patch.object(
            run_auto_trade, "PostgresPredictionResultRepository"
        ), patch(
            "src.orchestration.jobs.prediction_saved_marker.read_prediction_saved_day",
            return_value="20261030",
        ):
            rc = run_auto_trade.main(argv)
        return rc, captured

    def test_paper_passes_market_data(self):
        rc, captured = self._run([])

        self.assertEqual(rc, 0)
        self.assertIsNotNone(captured.get("market_data"))

    def test_order_error_returns_nonzero(self):
        rc, _ = self._run([], errors=1)

        self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
