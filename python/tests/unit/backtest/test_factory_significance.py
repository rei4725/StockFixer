"""factory_significance（ゲート用の統計量）の性質テスト。"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.backtest.factory_gate import portfolio_dsr
from src.backtest.factory_significance import (
    annualized_sharpe,
    per_period_sharpe,
    return_moments,
    sharpe_difference_z,
)
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

_DATES = pd.bdate_range("2024-01-01", periods=504)


def _series(mean: float, seed: int, sd: float = 0.01) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, len(_DATES)), index=_DATES)


class TestSharpe(unittest.TestCase):
    def test_annualizes_with_trading_days(self):
        r = _series(0.001, 0)
        self.assertAlmostEqual(annualized_sharpe(r), per_period_sharpe(r) * math.sqrt(252))

    def test_is_scale_invariant(self):
        """リターンを定数倍（レバレッジ）しても Sharpe は変わらない。"""
        r = _series(0.001, 0)
        self.assertAlmostEqual(per_period_sharpe(r), per_period_sharpe(r * 3))

    def test_nan_when_undefined(self):
        self.assertTrue(math.isnan(per_period_sharpe(pd.Series([0.0] * 10))))
        self.assertTrue(math.isnan(per_period_sharpe(pd.Series([0.01]))))


class TestSharpeDifferenceZ(unittest.TestCase):
    def test_positive_when_candidate_is_better(self):
        self.assertGreater(sharpe_difference_z(_series(0.003, 0), _series(0.0, 1)), 3)

    def test_is_antisymmetric(self):
        a, b = _series(0.002, 0), _series(0.0005, 1)
        self.assertAlmostEqual(sharpe_difference_z(a, b), -sharpe_difference_z(b, a))

    def test_equal_quality_is_rarely_significant(self):
        """同じ実力の 2 系列では z > 1.645 は約 5% しか起きない（片側検定の有意水準）。"""
        hits = sum(
            sharpe_difference_z(_series(0.001, 2 * s), _series(0.001, 2 * s + 1)) > 1.645
            for s in range(400)
        )
        self.assertLessEqual(hits / 400, 0.08)

    def test_uses_only_common_dates(self):
        a = _series(0.002, 0)
        b = _series(0.0, 1).iloc[100:]
        self.assertAlmostEqual(sharpe_difference_z(a, b), sharpe_difference_z(a.iloc[100:], b))

    def test_nan_when_not_testable(self):
        self.assertTrue(math.isnan(sharpe_difference_z(None, _series(0.0, 1))))
        a = _series(0.001, 0)
        # 同一系列は差の分散がゼロで検定できない
        self.assertTrue(math.isnan(sharpe_difference_z(a, a)))


class TestPortfolioDsr(unittest.TestCase):
    def _eval(self, returns):
        return FactoryEvaluation(
            hypothesis=FactoryHypothesis(
                rule_spec={"type": "atomic", "rule": "x", "params": {}}, market="jp"
            ),
            portfolio_returns=returns,
        )

    def test_noise_does_not_pass_even_with_many_trials(self):
        """期待値ゼロの戦略は DSR ゲート（0.95）を通らない。"""
        passed = sum(portfolio_dsr(self._eval(_series(0.0, s)), 1000) >= 0.95 for s in range(200))
        self.assertEqual(passed, 0)

    def test_does_not_saturate_for_moderate_edge(self):
        """年率 Sharpe 1.6 程度の戦略は、1000 試行の補正後に 1.0 へ張り付かない。

        取引数を観測数にしていた旧実装では、取引の多い戦略ほど DSR が 1.0 に張り付き
        多重比較の補正として機能していなかった。
        """
        dsr = portfolio_dsr(self._eval(_series(0.001, 0)), 1000)
        self.assertLess(dsr, 0.95)

    def test_nan_without_returns(self):
        self.assertTrue(math.isnan(portfolio_dsr(self._eval(None), 1000)))


class TestReturnMoments(unittest.TestCase):
    def test_normal_series_is_close_to_normal_moments(self):
        skew, kurt = return_moments(_series(0.0, 0))
        self.assertAlmostEqual(skew, 0.0, delta=0.3)
        self.assertAlmostEqual(kurt, 3.0, delta=0.5)

    def test_defaults_for_short_series(self):
        self.assertEqual(return_moments(pd.Series([0.1, 0.2])), (0.0, 3.0))


if __name__ == "__main__":
    unittest.main()
