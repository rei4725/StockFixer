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

    def test_calmar_is_none_when_no_drawdown(self):
        """equity が単調増加（max_drawdown == 0.0）のときは 0 除算を避け None を返す。"""
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertIn("sharpe_ratio", m)
        self.assertIsNone(m["calmar_ratio"])

    def test_sharpe_and_calmar_pinned_with_drawdown(self):
        """max_dd < 0（実際にドローダウンする equity）で calmar/sharpe を手計算値に固定する。

        equity: 1000 -> 1600(高値) -> 1200(安値) -> 1400(最終、部分回復)。
        pnls: [500, -100, 200]（3件の ClosedTrade.realized_pnl）。

        手計算（詳細は task-10-report.md の Fix レポートに記載）:
            years = (2024-12-31 - 2024-01-02).days / 365.25 = 364/365.25 = 0.9965776865...
            drawdown 系列 = [0, 0, (1200-1600)/1600, (1400-1600)/1600] = [0, 0, -0.25, -0.125]
            max_dd = -0.25
            cagr = (1400/1000)**(1/years) - 1 = 0.401618589... -> round(.,6) = 0.401619
            calmar_ratio = cagr / abs(max_dd) = 0.401619 / 0.25 = 1.606474... -> round(.,4) = 1.6065

            mean(pnls) = 200.0, std(pnls, ddof=1) = 300.0
            sharpe_per_trade = 200.0 / 300.0 = 0.666667
            trades_per_year = 3 / years = 3.010302...
            sharpe_ratio = sharpe_per_trade * sqrt(trades_per_year)
                         = 0.666667 * 1.735...  = 1.156681... -> round(.,4) = 1.1567
        """
        equity = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-04-01", "2024-07-01", "2024-12-31"],
                "portfolio_value": [1000.0, 1600.0, 1200.0, 1400.0],
            }
        )
        closed = [
            _trade("A", 500.0, 0.5, 1.5),
            _trade("B", -100.0, -0.2, 0.8),
            _trade("C", 200.0, 0.2, 1.2),
        ]
        trades = pd.DataFrame(
            [
                {"multiple": t.multiple, "max_multiple": t.max_multiple, "held_days": t.held_days}
                for t in closed
            ]
        )
        m = compute_longterm_metrics(equity, trades, _cfg(), {}, closed)
        self.assertAlmostEqual(m["calmar_ratio"], 1.6065, places=4)
        self.assertAlmostEqual(m["sharpe_ratio"], 1.1567, places=4)

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
