"""longterm メトリクスの損益対応と追加指標。"""

import unittest

import pandas as pd

from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.metrics import compute_longterm_metrics
from src.backtest.longterm.portfolio import ClosedTrade


def _trade(symbol, pnl, rate, multiple):
    return ClosedTrade(
        symbol=symbol,
        entry_date="2024-01-02",
        exit_date="2024-06-28",
        entry_price=10.0,
        exit_price=10.0 * multiple,
        multiple=multiple,
        max_multiple=multiple,
        exit_reason="ma_break",
        held_days=178,
        realized_pnl=pnl,
        return_rate=rate,
    )


def _cfg():
    return LongtermBacktestConfig.build(
        market="us", start="2024-01-02", end="2024-12-31", initial_cash=1000.0
    )


class TestLongtermMetrics(unittest.TestCase):
    def setUp(self):
        self.equity = pd.DataFrame(
            {"date": ["2024-01-02", "2024-06-28"], "portfolio_value": [1000.0, 1400.0]}
        )
        self.closed = [_trade("A", 500.0, 0.5, 1.5), _trade("B", -100.0, -0.2, 0.8)]
        self.trades = pd.DataFrame(
            [
                {"multiple": t.multiple, "max_multiple": t.max_multiple, "held_days": t.held_days}
                for t in self.closed
            ]
        )

    def test_key_is_num_trades(self):
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertEqual(m["num_trades"], 2)
        self.assertNotIn("n_trades", m)

    def test_profit_factor_from_realized_pnl(self):
        """銘柄ごとに正しく対になった損益から算出される（core の FIFO は使わない）。"""
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertAlmostEqual(m["profit_factor"], 5.0)

    def test_sharpe_and_calmar_present(self):
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertIn("sharpe_ratio", m)
        self.assertIn("calmar_ratio", m)

    def test_dsr_absent_when_no_trials(self):
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertNotIn("dsr", m)

    def test_multiple_buckets_preserved(self):
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertEqual(m["n_2x"], 0)
        self.assertIn("pct_2x", m)

    def test_empty_trades_does_not_crash(self):
        m = compute_longterm_metrics(
            self.equity,
            pd.DataFrame(columns=["multiple", "max_multiple", "held_days"]),
            _cfg(),
            {},
            [],
        )
        self.assertEqual(m["num_trades"], 0)
        self.assertAlmostEqual(m["profit_factor"], 0.0)


if __name__ == "__main__":
    unittest.main()
