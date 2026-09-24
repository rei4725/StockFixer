import math
import unittest

from src.backtest.slippage import _DEFAULT_ALPHA, estimate_slippage, make_slippage_fn


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
