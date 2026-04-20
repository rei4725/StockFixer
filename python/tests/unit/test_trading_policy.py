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


if __name__ == "__main__":
    unittest.main()
