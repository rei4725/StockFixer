"""
ユニットテスト: pair_trading

外部データ取得なし。numpy/pandas の合成データで検証する。
"""

import unittest

import numpy as np
import pandas as pd

from src.strategy.pair_trading import (
    PairBacktestResult,
    _adf_statistic,
    backtest_pair,
    check_cointegration,
    compute_spread,
    generate_pair_signals,
    run_pair_trading_scan,
    select_correlated_pairs,
)


def _make_cointegrated_prices(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """共和分する合成価格 DataFrame を生成する。"""
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.standard_normal(n)) + 100.0
    price_a = trend + rng.standard_normal(n) * 0.3
    price_b = 2.0 * trend + rng.standard_normal(n) * 0.3
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": price_a, "B": price_b}, index=idx)


class TestSelectCorrelatedPairs(unittest.TestCase):
    def test_high_correlation_pair_selected(self):
        rng = np.random.default_rng(0)
        n = 100
        base = np.cumsum(rng.standard_normal(n))
        prices = pd.DataFrame(
            {
                "A": base + rng.standard_normal(n) * 0.05,
                "B": base + rng.standard_normal(n) * 0.05,
                "C": np.cumsum(rng.standard_normal(n)),
            }
        )
        pairs = select_correlated_pairs(prices, corr_threshold=0.9)
        syms = [(a, b) for a, b, _ in pairs]
        self.assertIn(("A", "B"), syms)

    def test_no_pairs_below_threshold(self):
        rng = np.random.default_rng(1)
        prices = pd.DataFrame({"X": rng.standard_normal(50), "Y": rng.standard_normal(50)})
        pairs = select_correlated_pairs(prices, corr_threshold=0.99)
        self.assertEqual(pairs, [])

    def test_single_column_returns_empty(self):
        prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        self.assertEqual(select_correlated_pairs(prices), [])

    def test_result_sorted_by_abs_correlation(self):
        rng = np.random.default_rng(2)
        n = 100
        base = np.cumsum(rng.standard_normal(n))
        prices = pd.DataFrame(
            {
                "A": base + rng.standard_normal(n) * 0.05,
                "B": base + rng.standard_normal(n) * 0.1,
                "C": base + rng.standard_normal(n) * 0.2,
            }
        )
        pairs = select_correlated_pairs(prices, corr_threshold=0.8)
        if len(pairs) >= 2:
            corrs = [abs(c) for _, _, c in pairs]
            self.assertEqual(corrs, sorted(corrs, reverse=True))


class TestAdfStatistic(unittest.TestCase):
    def test_stationary_series_returns_negative(self):
        rng = np.random.default_rng(42)
        n = 300
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.3 * y[i - 1] + rng.standard_normal()
        stat = _adf_statistic(y)
        self.assertLess(stat, 0.0)

    def test_returns_finite_for_random_walk(self):
        rng = np.random.default_rng(7)
        y = np.cumsum(rng.standard_normal(200))
        stat = _adf_statistic(y)
        self.assertTrue(np.isfinite(stat))

    def test_short_series_returns_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        stat = _adf_statistic(y, max_lag=2)
        self.assertEqual(stat, 0.0)


class TestTestCointegration(unittest.TestCase):
    def test_returns_three_tuple(self):
        rng = np.random.default_rng(0)
        a = pd.Series(rng.standard_normal(50) + 100)
        b = pd.Series(rng.standard_normal(50) + 100)
        result = check_cointegration(a, b)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_insufficient_data_returns_false(self):
        a = pd.Series(np.arange(10, dtype=float))
        b = pd.Series(np.arange(10, dtype=float))
        _, _, is_coint = check_cointegration(a, b)
        self.assertFalse(is_coint)

    def test_cointegrated_pair_detected(self):
        rng = np.random.default_rng(42)
        n = 300
        trend = np.cumsum(rng.standard_normal(n))
        a = pd.Series(trend + rng.standard_normal(n) * 0.1)
        b = pd.Series(2.0 * trend + rng.standard_normal(n) * 0.1)
        _, adf_stat, is_coint = check_cointegration(a, b)
        self.assertIsInstance(is_coint, bool)
        self.assertIsInstance(adf_stat, float)
        self.assertTrue(is_coint, msg=f"ADF={adf_stat:.4f} — 共和分が検出されなかった")

    def test_hedge_ratio_is_float(self):
        rng = np.random.default_rng(3)
        n = 100
        trend = np.cumsum(rng.standard_normal(n))
        a = pd.Series(trend)
        b = pd.Series(trend + rng.standard_normal(n) * 0.5)
        hedge_ratio, _, _ = check_cointegration(a, b)
        self.assertIsInstance(hedge_ratio, float)


class TestComputeSpread(unittest.TestCase):
    def test_spread_is_mean_zero(self):
        rng = np.random.default_rng(0)
        n = 100
        a = pd.Series(np.cumsum(rng.standard_normal(n)))
        b = pd.Series(np.cumsum(rng.standard_normal(n)))
        spread = compute_spread(a, b, hedge_ratio=1.0)
        self.assertAlmostEqual(float(spread.mean()), 0.0, places=10)

    def test_spread_length_matches_input(self):
        n = 60
        a = pd.Series(np.ones(n))
        b = pd.Series(np.ones(n))
        spread = compute_spread(a, b, hedge_ratio=0.5)
        self.assertEqual(len(spread), n)


class TestGeneratePairSignals(unittest.TestCase):
    def test_output_has_correct_columns(self):
        spread = pd.Series(np.zeros(50))
        signals = generate_pair_signals(spread)
        self.assertIn("position_a", signals.columns)
        self.assertIn("position_b", signals.columns)

    def test_positions_are_opposite(self):
        t = np.linspace(0, 4 * np.pi, 200)
        spread = pd.Series(np.sin(t) * 3)
        signals = generate_pair_signals(spread, entry_z=1.5, exit_z=0.3, window=10)
        for _, row in signals.iterrows():
            if row["position_a"] != 0.0:
                self.assertAlmostEqual(row["position_a"], -row["position_b"])

    def test_no_signal_when_spread_flat(self):
        spread = pd.Series(np.zeros(100))
        signals = generate_pair_signals(spread)
        self.assertTrue((signals["position_a"] == 0.0).all())


class TestBacktestPair(unittest.TestCase):
    def test_missing_symbol_returns_none(self):
        prices = _make_cointegrated_prices()
        result = backtest_pair(prices, "A", "NONEXISTENT")
        self.assertIsNone(result)

    def test_insufficient_data_returns_none(self):
        rng = np.random.default_rng(0)
        prices = pd.DataFrame({"A": rng.standard_normal(10), "B": rng.standard_normal(10)})
        result = backtest_pair(prices, "A", "B")
        self.assertIsNone(result)

    def test_result_fields_populated(self):
        prices = _make_cointegrated_prices()
        result = backtest_pair(prices, "A", "B")
        if result is not None:
            self.assertIsInstance(result, PairBacktestResult)
            self.assertEqual(result.symbol_a, "A")
            self.assertEqual(result.symbol_b, "B")
            self.assertIsInstance(result.total_return, float)
            self.assertIsInstance(result.sharpe_ratio, float)
            self.assertIsInstance(result.num_trades, int)
            self.assertGreaterEqual(result.num_trades, 0)
            self.assertLessEqual(result.max_drawdown, 0.0)

    def test_win_rate_in_range(self):
        prices = _make_cointegrated_prices()
        result = backtest_pair(prices, "A", "B")
        if result is not None and result.num_trades > 0:
            self.assertGreaterEqual(result.win_rate, 0.0)
            self.assertLessEqual(result.win_rate, 1.0)


class TestRunPairTradingScan(unittest.TestCase):
    def test_returns_list(self):
        prices = _make_cointegrated_prices()
        results = run_pair_trading_scan(prices, corr_threshold=0.8)
        self.assertIsInstance(results, list)

    def test_single_column_returns_empty(self):
        prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        results = run_pair_trading_scan(prices)
        self.assertEqual(results, [])

    def test_results_sorted_by_sharpe(self):
        rng = np.random.default_rng(99)
        n = 200
        trend = np.cumsum(rng.standard_normal(n)) + 100.0
        prices = pd.DataFrame(
            {
                "A": trend + rng.standard_normal(n) * 0.3,
                "B": 2.0 * trend + rng.standard_normal(n) * 0.3,
                "C": 0.5 * trend + rng.standard_normal(n) * 0.3,
            }
        )
        results = run_pair_trading_scan(prices, corr_threshold=0.8)
        if len(results) >= 2:
            sharpes = [r.sharpe_ratio for r in results]
            self.assertEqual(sharpes, sorted(sharpes, reverse=True))
