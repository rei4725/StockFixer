import math
import unittest
from unittest.mock import MagicMock

from src.backtest.slippage import (
    _DEFAULT_ALPHA,
    calibrate_alpha,
    estimate_slippage,
    get_calibrated_slippage_fn,
    make_slippage_fn,
)


class TestEstimateSlippage(unittest.TestCase):
    def test_zero_volume_returns_zero(self):
        self.assertEqual(estimate_slippage(100, 1000.0, 0), 0.0)

    def test_zero_qty_returns_zero(self):
        self.assertEqual(estimate_slippage(0, 1000.0, 50000), 0.0)

    def test_zero_price_returns_zero(self):
        self.assertEqual(estimate_slippage(100, 0.0, 50000), 0.0)

    def test_negative_volume_returns_zero(self):
        self.assertEqual(estimate_slippage(100, 1000.0, -1), 0.0)

    def test_formula_sqrt_participation_rate(self):
        alpha = 0.10
        qty = 200
        price = 1500.0
        adv = 50000
        expected = alpha * math.sqrt(qty / adv)
        result = estimate_slippage(qty, price, adv, alpha=alpha)
        self.assertAlmostEqual(result, expected, places=8)

    def test_custom_alpha(self):
        result = estimate_slippage(100, 1000.0, 10000, alpha=0.05)
        expected = 0.05 * math.sqrt(100 / 10000)
        self.assertAlmostEqual(result, expected, places=8)

    def test_large_order_larger_slippage(self):
        small = estimate_slippage(100, 1000.0, 50000)
        large = estimate_slippage(5000, 1000.0, 50000)
        self.assertGreater(large, small)

    def test_returns_float(self):
        result = estimate_slippage(100, 1000.0, 50000)
        self.assertIsInstance(result, float)


class TestMakeSlippageFn(unittest.TestCase):
    def test_returns_callable(self):
        fn = make_slippage_fn()
        self.assertTrue(callable(fn))

    def test_fn_uses_alpha(self):
        alpha = 0.05
        fn = make_slippage_fn(alpha=alpha)
        result = fn(100, 1000.0, 10000)
        expected = estimate_slippage(100, 1000.0, 10000, alpha=alpha)
        self.assertAlmostEqual(result, expected, places=8)

    def test_default_alpha_used_when_omitted(self):
        fn = make_slippage_fn()
        result = fn(100, 1000.0, 10000)
        expected = estimate_slippage(100, 1000.0, 10000, alpha=_DEFAULT_ALPHA)
        self.assertAlmostEqual(result, expected, places=8)

    def test_fn_zero_qty_returns_zero(self):
        fn = make_slippage_fn()
        self.assertEqual(fn(0, 1000.0, 10000), 0.0)


class TestCalibrateAlpha(unittest.TestCase):
    def _make_con(self, rows=None, raise_error=False):
        con = MagicMock()
        if raise_error:
            con.execute.side_effect = Exception("table not found")
        else:
            con.execute.return_value.fetchall.return_value = rows or []
        return con

    def test_returns_default_on_exception(self):
        con = self._make_con(raise_error=True)
        result = calibrate_alpha(con)
        self.assertEqual(result, _DEFAULT_ALPHA)

    def test_returns_default_when_too_few_samples(self):
        con = self._make_con(rows=[(1000.0, 1001.0, 1)])
        result = calibrate_alpha(con, min_samples=10)
        self.assertEqual(result, _DEFAULT_ALPHA)

    def test_computes_alpha_from_buy_slippage(self):
        rows = [(1000.0, 1010.0, 1)] * 15
        con = self._make_con(rows=rows)
        result = calibrate_alpha(con, min_samples=10)
        self.assertGreater(result, 0.0)
        self.assertLessEqual(result, 0.05)

    def test_computes_alpha_from_sell_slippage(self):
        rows = [(1010.0, 1000.0, 2)] * 15
        con = self._make_con(rows=rows)
        result = calibrate_alpha(con, min_samples=10)
        self.assertGreater(result, 0.0)

    def test_negative_slippages_excluded_returns_default(self):
        rows = [(1010.0, 1000.0, 1)] * 15
        con = self._make_con(rows=rows)
        result = calibrate_alpha(con, min_samples=10)
        self.assertEqual(result, _DEFAULT_ALPHA)

    def test_alpha_clipped_to_max(self):
        rows = [(1000.0, 2000.0, 1)] * 20
        con = self._make_con(rows=rows)
        result = calibrate_alpha(con, min_samples=10)
        self.assertLessEqual(result, 0.05)

    def test_alpha_clipped_to_min(self):
        rows = [(1000.0, 1000.0, 1)] * 20
        con = self._make_con(rows=rows)
        result = calibrate_alpha(con, min_samples=10)
        self.assertGreaterEqual(result, 0.0)


class TestGetCalibratedSlippageFn(unittest.TestCase):
    def test_returns_callable(self):
        con = MagicMock()
        con.execute.return_value.fetchall.return_value = []
        fn = get_calibrated_slippage_fn(con)
        self.assertTrue(callable(fn))

    def test_fn_returns_float(self):
        con = MagicMock()
        con.execute.return_value.fetchall.return_value = []
        fn = get_calibrated_slippage_fn(con)
        result = fn(100, 1000.0, 50000)
        self.assertIsInstance(result, float)
