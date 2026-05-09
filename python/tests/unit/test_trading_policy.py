"""
ユニットテスト: config.trading_policy
"""
import importlib
import unittest
from unittest.mock import patch


class TestTradingPolicyDefaults(unittest.TestCase):
    def _load(self, env: dict):
        import config.trading_policy as tp

        with patch.dict("os.environ", env, clear=False):
            importlib.reload(tp)
            return tp

    def test_default_profile_is_moderate(self):
        tp = self._load({})
        self.assertEqual(tp.RISK_PROFILE, "moderate")

    def test_moderate_defaults(self):
        tp = self._load({})
        self.assertAlmostEqual(tp.MAX_ACCEPTABLE_DRAWDOWN, 0.20)
        self.assertAlmostEqual(tp.KELLY_CAP, 0.75)
        self.assertAlmostEqual(tp.HIGH_CONFIDENCE_POSITION_CAP, 0.30)
        self.assertAlmostEqual(tp.MIN_SHARPE_TO_TRADE, 0.50)


class TestTradingPolicyProfileOverride(unittest.TestCase):
    def _load(self, env: dict):
        import config.trading_policy as tp

        with patch.dict("os.environ", env, clear=False):
            importlib.reload(tp)
            return tp

    def test_conservative_profile(self):
        tp = self._load({"RISK_PROFILE": "conservative"})
        self.assertEqual(tp.RISK_PROFILE, "conservative")
        self.assertAlmostEqual(tp.MAX_ACCEPTABLE_DRAWDOWN, 0.10)
        self.assertAlmostEqual(tp.KELLY_CAP, 0.50)
        self.assertAlmostEqual(tp.HIGH_CONFIDENCE_POSITION_CAP, 0.20)
        self.assertAlmostEqual(tp.MIN_SHARPE_TO_TRADE, 0.80)

    def test_aggressive_profile(self):
        tp = self._load({"RISK_PROFILE": "aggressive"})
        self.assertEqual(tp.RISK_PROFILE, "aggressive")
        self.assertAlmostEqual(tp.MAX_ACCEPTABLE_DRAWDOWN, 0.30)
        self.assertAlmostEqual(tp.KELLY_CAP, 1.00)
        self.assertAlmostEqual(tp.HIGH_CONFIDENCE_POSITION_CAP, 0.40)
        self.assertAlmostEqual(tp.MIN_SHARPE_TO_TRADE, 0.30)

    def test_invalid_profile_raises_value_error(self):
        import config.trading_policy as tp

        with patch.dict("os.environ", {"RISK_PROFILE": "unknown"}, clear=False):
            with self.assertRaises(ValueError):
                importlib.reload(tp)

    def tearDown(self):
        import config.trading_policy as _tp

        with patch.dict("os.environ", {"RISK_PROFILE": "moderate"}, clear=False):
            try:
                importlib.reload(_tp)
            except Exception:
                pass


class TestTradingPolicyIndividualOverride(unittest.TestCase):
    def _load(self, env: dict):
        import config.trading_policy as tp

        with patch.dict("os.environ", env, clear=False):
            importlib.reload(tp)
            return tp

    def test_kelly_cap_individual_override(self):
        tp = self._load({"RISK_PROFILE": "moderate", "KELLY_CAP": "0.9"})
        self.assertAlmostEqual(tp.KELLY_CAP, 0.9)

    def test_invalid_kelly_cap_raises_value_error(self):
        import config.trading_policy as tp

        with patch.dict("os.environ", {"KELLY_CAP": "not_a_number"}, clear=False):
            with self.assertRaises(ValueError):
                importlib.reload(tp)

    def test_kelly_cap_out_of_range_raises_value_error(self):
        import config.trading_policy as tp

        with patch.dict("os.environ", {"KELLY_CAP": "1.5"}, clear=False):
            with self.assertRaises(ValueError):
                importlib.reload(tp)

    def test_kelly_cap_nan_raises_value_error(self):
        import config.trading_policy as tp

        with patch.dict("os.environ", {"KELLY_CAP": "nan"}, clear=False):
            with self.assertRaises(ValueError):
                importlib.reload(tp)

    def tearDown(self):
        import config.trading_policy as _tp

        with patch.dict("os.environ", {"RISK_PROFILE": "moderate"}, clear=False):
            try:
                importlib.reload(_tp)
            except Exception:
                pass


class TestVixHedgeDefaults(unittest.TestCase):
    """R-406: VIX テールリスクヘッジパラメータのデフォルト値テスト"""

    def _load(self, env: dict):
        import config.trading_policy as tp

        with patch.dict("os.environ", env, clear=False):
            importlib.reload(tp)
            return tp

    def test_vix_spike_threshold_default(self):
        tp = self._load({})
        self.assertAlmostEqual(tp.VIX_SPIKE_THRESHOLD, 30.0)

    def test_vix_position_scale_default(self):
        tp = self._load({})
        self.assertAlmostEqual(tp.VIX_POSITION_SCALE, 0.5)

    def test_vix_hedge_enabled_default_false(self):
        tp = self._load({})
        self.assertFalse(tp.VIX_HEDGE_ENABLED)

    def test_vix_spike_threshold_env_override(self):
        tp = self._load({"VIX_SPIKE_THRESHOLD": "40.0"})
        self.assertAlmostEqual(tp.VIX_SPIKE_THRESHOLD, 40.0)

    def test_vix_position_scale_env_override(self):
        tp = self._load({"VIX_POSITION_SCALE": "0.3"})
        self.assertAlmostEqual(tp.VIX_POSITION_SCALE, 0.3)

    def test_vix_hedge_enabled_via_env(self):
        tp = self._load({"VIX_HEDGE_ENABLED": "true"})
        self.assertTrue(tp.VIX_HEDGE_ENABLED)

    def test_vix_spike_threshold_invalid_raises(self):
        import config.trading_policy as tp

        with patch.dict("os.environ", {"VIX_SPIKE_THRESHOLD": "not_a_number"}, clear=False):
            with self.assertRaises(ValueError):
                importlib.reload(tp)

    def tearDown(self):
        import config.trading_policy as _tp

        with patch.dict("os.environ", {"RISK_PROFILE": "moderate"}, clear=False):
            try:
                importlib.reload(_tp)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
