"""backtest/metrics の取引損益をコスト込みで数えるテスト（#953）

net の勝率・PF・Sharpe は手数料とスリッページを引いた後の損益で、
gross は値段の差のままで出す。ショート（short → short_cover 系）も取引に数える。
"""

import unittest

import pandas as pd

from src.backtest.metrics import (
    _extract_trade_pnl,
    compute_cost_comparison_metrics,
    compute_metrics,
)


def _costly_log() -> pd.DataFrame:
    """片道 1% の手数料がかかる 3 取引のログ（初期資金 10,000）

    1: 100 → 101  gross +10 / net -10（コストで負けに転じる）
    2: 100 → 120  gross +200 / net +180
    3: 100 → 90   gross -100 / net -120
    """
    return pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "action": "buy",
                "price": 100.0,
                "qty": 10,
                "cash": 8_990.0,
                "cash_gross": 9_000.0,
            },
            {
                "date": "2024-01-10",
                "action": "sell",
                "price": 101.0,
                "qty": 10,
                "cash": 9_990.0,
                "cash_gross": 10_010.0,
            },
            {
                "date": "2024-02-01",
                "action": "buy",
                "price": 100.0,
                "qty": 10,
                "cash": 8_980.0,
                "cash_gross": 9_010.0,
            },
            {
                "date": "2024-02-10",
                "action": "sell",
                "price": 120.0,
                "qty": 10,
                "cash": 10_170.0,
                "cash_gross": 10_210.0,
            },
            {
                "date": "2024-03-01",
                "action": "buy",
                "price": 100.0,
                "qty": 10,
                "cash": 9_160.0,
                "cash_gross": 9_210.0,
            },
            {
                "date": "2024-03-10",
                "action": "stop_loss",
                "price": 90.0,
                "qty": 10,
                "cash": 10_050.0,
                "cash_gross": 10_110.0,
            },
        ]
    )


class TestNetOfCostTradePnl(unittest.TestCase):
    """net の取引損益は手数料とスリッページを引いた後の値になる"""

    def test_net_pnl_subtracts_costs(self):
        wins, losses, win_returns, loss_returns = _extract_trade_pnl(
            _costly_log(), net_of_costs=True
        )
        self.assertEqual(len(wins), 1)
        self.assertAlmostEqual(wins[0], 180.0)
        self.assertEqual(len(losses), 2)
        self.assertAlmostEqual(losses[0], -10.0)
        self.assertAlmostEqual(losses[1], -120.0)
        self.assertAlmostEqual(win_returns[0], 0.18)
        self.assertAlmostEqual(loss_returns[0], 0.01)

    def test_gross_pnl_is_price_difference(self):
        wins, losses = _extract_trade_pnl(_costly_log())[:2]
        self.assertEqual([round(w, 6) for w in wins], [10.0, 200.0])
        self.assertEqual([round(x, 6) for x in losses], [-100.0])

    def test_net_metrics_differ_from_gross(self):
        log = _costly_log()
        net = compute_metrics(log, 10_000)
        gross = compute_metrics(log, 10_000, cash_column="cash_gross")
        # コストで負けに転じた取引 1 が負けに数えられる
        self.assertAlmostEqual(net["win_rate"], 0.3333)
        self.assertAlmostEqual(gross["win_rate"], 0.6667)
        self.assertAlmostEqual(net["profit_factor"], round(180.0 / 130.0, 4))
        self.assertAlmostEqual(gross["profit_factor"], 2.1)
        self.assertNotEqual(net["sharpe_ratio"], gross["sharpe_ratio"])
        self.assertAlmostEqual(net["avg_win"], 0.18)
        self.assertAlmostEqual(net["avg_loss"], 0.065)
        self.assertAlmostEqual(gross["avg_win"], 0.105)
        self.assertAlmostEqual(gross["avg_loss"], 0.1)

    def test_cost_comparison_net_sharpe_differs_from_gross(self):
        result = compute_cost_comparison_metrics(_costly_log(), 10_000)
        self.assertAlmostEqual(result["win_rate"], 0.3333)
        self.assertNotEqual(result["sharpe_ratio"], result["gross_sharpe_ratio"])
        self.assertLess(result["sharpe_ratio"], result["gross_sharpe_ratio"])

    def test_zero_cost_net_equals_gross(self):
        log = _costly_log()
        log["cash"] = log["cash_gross"]
        net = compute_metrics(log, 10_000)
        gross = compute_metrics(log, 10_000, cash_column="cash_gross")
        for key in ("win_rate", "profit_factor", "sharpe_ratio", "avg_win", "avg_loss"):
            self.assertEqual(net[key], gross[key], key)


class TestShortTradePnl(unittest.TestCase):
    """short → short_cover 系の対も取引に数える"""

    def _log(self) -> pd.DataFrame:
        # ショート 100 → 90（+100）と、ロング 100 → 95（-50）。コストなし。
        return pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "action": "short",
                    "price": 100.0,
                    "qty": 10,
                    "cash": 10_000.0,
                    "cash_gross": 10_000.0,
                },
                {
                    "date": "2024-01-10",
                    "action": "short_cover",
                    "price": 90.0,
                    "qty": 10,
                    "cash": 10_100.0,
                    "cash_gross": 10_100.0,
                },
                {
                    "date": "2024-02-01",
                    "action": "buy",
                    "price": 100.0,
                    "qty": 10,
                    "cash": 9_100.0,
                    "cash_gross": 9_100.0,
                },
                {
                    "date": "2024-02-10",
                    "action": "sell",
                    "price": 95.0,
                    "qty": 10,
                    "cash": 10_050.0,
                    "cash_gross": 10_050.0,
                },
            ]
        )

    def test_short_counted_in_num_trades_and_win_rate(self):
        metrics = compute_metrics(self._log(), 10_000)
        self.assertEqual(metrics["num_trades"], 2)
        self.assertAlmostEqual(metrics["win_rate"], 0.5)

    def test_short_pnl_and_return(self):
        wins, losses, win_returns, _ = _extract_trade_pnl(self._log())
        self.assertAlmostEqual(wins[0], 100.0)
        self.assertAlmostEqual(win_returns[0], 0.1)
        self.assertAlmostEqual(losses[0], -50.0)

    def test_final_short_cover_net_of_costs(self):
        # ショート 100 → 99 は gross +10 だが、片道 1% のコストで net -10
        log = pd.DataFrame(
            [
                {
                    "date": "2024-01-02",
                    "action": "short",
                    "price": 100.0,
                    "qty": 10,
                    "cash": 10_000.0,
                    "cash_gross": 10_000.0,
                },
                {
                    "date": "2024-01-10",
                    "action": "final_short_cover",
                    "price": 99.0,
                    "qty": 10,
                    "cash": 9_990.0,
                    "cash_gross": 10_010.0,
                },
            ]
        )
        net_wins, net_losses = _extract_trade_pnl(log, net_of_costs=True)[:2]
        self.assertEqual(net_wins, [])
        self.assertAlmostEqual(net_losses[0], -10.0)
        gross_wins = _extract_trade_pnl(log)[0]
        self.assertAlmostEqual(gross_wins[0], 10.0)


if __name__ == "__main__":
    unittest.main()
