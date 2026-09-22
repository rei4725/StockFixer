"""BacktestScreeningPort の注入と未注入エラー。"""

import unittest

import pandas as pd

from src.backtest import screening_port as sp


class _FakePort:
    def screen_trend_candidates(self, market, top_n, as_of):
        return []

    def simulate_position(self, prices, entry_date, rules):
        return []


class TestPortInjection(unittest.TestCase):
    def setUp(self):
        self._saved = sp._port

    def tearDown(self):
        sp._port = self._saved

    def test_raises_when_not_injected(self):
        sp._port = None
        with self.assertRaises(RuntimeError) as ctx:
            sp.get_backtest_screening_port()
        self.assertIn("wire_ports", str(ctx.exception))

    def test_returns_injected_instance(self):
        fake = _FakePort()
        sp.set_backtest_screening_port(fake)
        self.assertIs(sp.get_backtest_screening_port(), fake)

    def test_fake_satisfies_protocol(self):
        self.assertIsInstance(_FakePort(), sp.BacktestScreeningPort)

    def test_object_without_methods_does_not_satisfy_protocol(self):
        self.assertNotIsInstance(object(), sp.BacktestScreeningPort)

    def test_injected_port_is_callable_through_getter(self):
        sp.set_backtest_screening_port(_FakePort())
        port = sp.get_backtest_screening_port()
        self.assertEqual(port.screen_trend_candidates("us", 5, "2024-01-02"), [])
        self.assertEqual(
            port.simulate_position(pd.DataFrame({"date": [], "Close": []}), "2024-01-02", None), []
        )


if __name__ == "__main__":
    unittest.main()
