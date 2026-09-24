"""TradeLedger の記録と再構成。"""

import unittest

from src.backtest.longterm.ledger import TradeLedger


class TestLedger(unittest.TestCase):
    def test_records_in_order(self):
        led = TradeLedger()
        led.record_buy("2024-01-02", "A", 10.0, 100, 1001.0, 8999.0)
        led.record_sell("2024-02-02", "A", 12.0, 100, 1198.0, 10197.0)
        df = led.to_frame()
        self.assertEqual(list(df["action"]), ["buy", "sell"])
        self.assertEqual(list(df["symbol"]), ["A", "A"])
        self.assertEqual(
            list(df.columns),
            ["date", "action", "symbol", "price", "qty", "amount", "cash"],
        )

    def test_net_cash_flow_matches_cash_delta(self):
        """買い支払 − 売り受取 が現金の減少分と一致する。"""
        led = TradeLedger()
        led.record_buy("2024-01-02", "A", 10.0, 100, 1001.0, 8999.0)
        led.record_sell("2024-02-02", "A", 12.0, 100, 1198.0, 10197.0)
        self.assertAlmostEqual(led.net_cash_flow(), 1001.0 - 1198.0)

    def test_scale_out_action_label(self):
        led = TradeLedger()
        led.record_sell("2024-02-02", "A", 12.0, 20, 239.0, 239.0, action="scale_out")
        self.assertEqual(led.to_frame()["action"].iloc[0], "scale_out")

    def test_empty_frame_has_columns(self):
        df = TradeLedger().to_frame()
        self.assertTrue(df.empty)
        self.assertIn("cash", df.columns)


if __name__ == "__main__":
    unittest.main()
