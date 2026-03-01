"""
backtest/metrics.py のユニットテスト
"""
import math
import unittest
import pandas as pd
import numpy as np
from src.backtest.metrics import (
    compute_metrics,
    _empty_metrics,
    _extract_trade_pnl,
    _sharpe_ratio,
    _max_drawdown,
)


def _trade_log(entries: list[dict]) -> pd.DataFrame:
    """テスト用の取引ログ DataFrame を生成する"""
    return pd.DataFrame(entries)


class TestComputeMetricsEmpty(unittest.TestCase):
    """取引なしのケース"""

    def test_empty_dataframe_returns_zero_metrics(self):
        metrics = compute_metrics(pd.DataFrame(), initial_cash=1_000_000)
        self.assertEqual(metrics["total_return"], 0.0)
        self.assertEqual(metrics["num_trades"], 0)
        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["sharpe_ratio"], 0.0)
        self.assertEqual(metrics["max_drawdown"], 0.0)
        self.assertIsNone(metrics["profit_factor"])

    def test_none_returns_empty_metrics(self):
        metrics = compute_metrics(None, initial_cash=500_000)
        self.assertEqual(metrics["final_cash"], 500_000)
        self.assertEqual(metrics["num_trades"], 0)


class TestComputeMetricsProfitable(unittest.TestCase):
    """利益トレードのケース"""

    def setUp(self):
        # 100株 @ 100円で買い、110円で売り → 利益 1000円
        self.initial_cash = 1_000_000
        self.log = _trade_log([
            {"date": "2024-01-02", "action": "buy",  "price": 100.0, "qty": 100, "cash": 990_000},
            {"date": "2024-01-10", "action": "sell", "price": 110.0, "qty": 100, "cash": 1_001_000},
        ])

    def test_num_trades(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertEqual(metrics["num_trades"], 1)

    def test_win_rate_one(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertAlmostEqual(metrics["win_rate"], 1.0)

    def test_profit_factor_infinite_or_positive(self):
        """勝ちトレードのみ → profit_factor は None(inf) または正の値"""
        metrics = compute_metrics(self.log, self.initial_cash)
        # 損失ゼロなので inf → None に変換
        self.assertIsNone(metrics["profit_factor"])

    def test_total_return_positive(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertGreater(metrics["total_return"], 0.0)


class TestComputeMetricsLoss(unittest.TestCase):
    """損失トレードのケース"""

    def setUp(self):
        self.initial_cash = 1_000_000
        self.log = _trade_log([
            {"date": "2024-01-02", "action": "buy",  "price": 100.0, "qty": 100, "cash": 990_000},
            {"date": "2024-01-10", "action": "sell", "price":  90.0, "qty": 100, "cash": 999_000},
        ])

    def test_win_rate_zero(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertAlmostEqual(metrics["win_rate"], 0.0)

    def test_num_trades_one(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertEqual(metrics["num_trades"], 1)


class TestComputeMetricsMixed(unittest.TestCase):
    """勝ち負け混在のケース"""

    def setUp(self):
        self.initial_cash = 1_000_000
        # 1回目: 100→120 (+2000), 2回目: 100→80 (-2000)
        self.log = _trade_log([
            {"date": "2024-01-02", "action": "buy",  "price": 100.0, "qty": 100, "cash": 990_000},
            {"date": "2024-01-05", "action": "sell", "price": 120.0, "qty": 100, "cash": 1_002_000},
            {"date": "2024-01-07", "action": "buy",  "price": 100.0, "qty": 100, "cash": 992_000},
            {"date": "2024-01-10", "action": "sell", "price":  80.0, "qty": 100, "cash": 1_000_000},
        ])

    def test_num_trades_two(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertEqual(metrics["num_trades"], 2)

    def test_win_rate_half(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertAlmostEqual(metrics["win_rate"], 0.5)

    def test_profit_factor_positive(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertIsNotNone(metrics["profit_factor"])
        self.assertGreater(metrics["profit_factor"], 0)

    def test_sharpe_ratio_is_float(self):
        metrics = compute_metrics(self.log, self.initial_cash)
        self.assertIsInstance(metrics["sharpe_ratio"], float)


class TestExtractTradePnl(unittest.TestCase):
    """_extract_trade_pnl のテスト"""

    def test_single_win(self):
        log = _trade_log([
            {"action": "buy",  "price": 100.0, "qty": 10, "cash": 900},
            {"action": "sell", "price": 110.0, "qty": 10, "cash": 1000},
        ])
        wins, losses = _extract_trade_pnl(log)
        self.assertEqual(len(wins), 1)
        self.assertEqual(len(losses), 0)
        self.assertAlmostEqual(wins[0], 100.0)

    def test_single_loss(self):
        log = _trade_log([
            {"action": "buy",  "price": 100.0, "qty": 10, "cash": 900},
            {"action": "sell", "price":  90.0, "qty": 10, "cash": 800},
        ])
        wins, losses = _extract_trade_pnl(log)
        self.assertEqual(len(wins), 0)
        self.assertEqual(len(losses), 1)
        self.assertAlmostEqual(losses[0], -100.0)

    def test_no_trades(self):
        wins, losses = _extract_trade_pnl(pd.DataFrame())
        self.assertEqual(wins, [])
        self.assertEqual(losses, [])

    def test_final_sell_counted(self):
        log = _trade_log([
            {"action": "buy",        "price": 100.0, "qty": 5, "cash": 950},
            {"action": "final_sell", "price": 120.0, "qty": 5, "cash": 1050},
        ])
        wins, losses = _extract_trade_pnl(log)
        self.assertEqual(len(wins), 1)


class TestSharpeRatio(unittest.TestCase):
    """_sharpe_ratio のテスト"""

    def test_empty_list_returns_zero(self):
        self.assertEqual(_sharpe_ratio([], 0.0, 252), 0.0)

    def test_single_item_returns_zero(self):
        self.assertEqual(_sharpe_ratio([100.0], 0.0, 252), 0.0)

    def test_zero_std_returns_zero(self):
        self.assertEqual(_sharpe_ratio([100.0, 100.0, 100.0], 0.0, 252), 0.0)

    def test_positive_returns_positive_sharpe(self):
        pnl = [200.0, 100.0, 300.0, 150.0, 250.0]
        sharpe = _sharpe_ratio(pnl, 0.0, 252)
        self.assertGreater(sharpe, 0.0)

    def test_negative_returns_negative_sharpe(self):
        pnl = [-200.0, -100.0, -300.0, -150.0, -250.0]
        sharpe = _sharpe_ratio(pnl, 0.0, 252)
        self.assertLess(sharpe, 0.0)


class TestMaxDrawdown(unittest.TestCase):
    """_max_drawdown のテスト"""

    def test_monotone_increase_no_drawdown(self):
        equity = pd.Series([1000, 1100, 1200, 1300])
        dd = _max_drawdown(equity)
        self.assertAlmostEqual(dd, 0.0)

    def test_monotone_decrease_full_drawdown(self):
        equity = pd.Series([1000, 900, 800, 700])
        dd = _max_drawdown(equity)
        self.assertLess(dd, 0.0)

    def test_drawdown_peak_then_recover(self):
        # 1000→1200→900→1100: max DD = (900-1200)/1200 = -0.25
        equity = pd.Series([1000.0, 1200.0, 900.0, 1100.0])
        dd = _max_drawdown(equity)
        self.assertAlmostEqual(dd, (900 - 1200) / 1200, places=8)

    def test_empty_series_returns_zero(self):
        dd = _max_drawdown(pd.Series([], dtype=float))
        self.assertEqual(dd, 0.0)


if __name__ == "__main__":
    unittest.main()
