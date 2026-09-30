"""build_evaluation（銘柄別結果 → 仮説単位の評価）のテスト。"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.backtest.factory_aggregation import SymbolMetrics
from src.backtest.factory_evaluation import build_evaluation
from src.backtest.types import FactoryHypothesis

_HYPOTHESIS = FactoryHypothesis(
    rule_spec={"type": "atomic", "rule": "rsi_contrarian", "params": {}}, market="jp"
)
_CASH = 100.0
_DATES = pd.bdate_range("2024-01-01", periods=6)


def _row(symbol: str, num_trades: int) -> SymbolMetrics:
    return SymbolMetrics(
        symbol=symbol,
        num_trades=num_trades,
        sharpe_ratio=1.0,
        sharpe_per_trade=0.1,
        win_rate=0.5,
        total_return=0.0,
        max_drawdown=-0.1,
        trade_returns=[1.0, -0.5, 2.0][:num_trades] + [0.3] * max(num_trades - 3, 0),
    )


def _curve(values: list[float]) -> pd.Series:
    return pd.Series(values, index=_DATES, dtype=float)


class TestBuildEvaluation(unittest.TestCase):
    def _build(self, rows, equity):
        return build_evaluation(
            _HYPOTHESIS,
            rows,
            equity,
            initial_cash=_CASH,
            min_trades_per_symbol=3,
            n_symbols=len(rows),
            window_returns=[0.1, 0.2],
        )

    def test_portfolio_metrics_come_from_equal_weight_daily_equity(self):
        equity = {
            "AAA": _curve([100, 102, 101, 103, 104, 106]),
            "BBB": _curve([100, 99, 101, 100, 102, 103]),
        }
        ev = self._build([_row("AAA", 3), _row("BBB", 4)], equity)

        portfolio = (equity["AAA"] + equity["BBB"]) / 2 / _CASH
        expected_returns = portfolio.pct_change().iloc[1:]
        pd.testing.assert_series_equal(
            ev.portfolio_returns, expected_returns, check_names=False, check_freq=False
        )
        expected_sharpe = expected_returns.mean() / expected_returns.std() * math.sqrt(252)
        self.assertAlmostEqual(ev.portfolio_sharpe_ratio, expected_sharpe)
        running_max = portfolio.cummax()
        self.assertAlmostEqual(
            ev.portfolio_max_drawdown, float((portfolio / running_max - 1).min())
        )
        self.assertEqual(ev.window_returns, [0.1, 0.2])

    def test_ineffective_symbols_are_excluded_from_portfolio(self):
        """最低取引数に満たない銘柄は集計にもポートフォリオにも入れない（#625 と同じ母集団）。"""
        equity = {
            "AAA": _curve([100, 102, 101, 103, 104, 106]),
            "FEW": _curve([100, 150, 60, 150, 60, 150]),  # 2 取引だけの暴れる銘柄
        }
        ev = self._build([_row("AAA", 3), _row("FEW", 2)], equity)

        expected = (equity["AAA"] / _CASH).pct_change().iloc[1:]
        pd.testing.assert_series_equal(
            ev.portfolio_returns, expected, check_names=False, check_freq=False
        )
        self.assertEqual(ev.n_effective_symbols, 1)

    def test_metrics_are_nan_without_equity(self):
        ev = self._build([_row("AAA", 3)], {})

        self.assertIsNone(ev.portfolio_returns)
        self.assertTrue(math.isnan(ev.portfolio_sharpe_ratio))
        self.assertTrue(math.isnan(ev.portfolio_max_drawdown))

    def test_sharpe_is_nan_for_flat_equity(self):
        """保有しても値動きが無い（分散ゼロ）なら Sharpe は算出不能。"""
        ev = self._build([_row("AAA", 3)], {"AAA": _curve([100.0] * 6)})

        self.assertTrue(math.isnan(ev.portfolio_sharpe_ratio))
        self.assertTrue(np.allclose(ev.portfolio_returns.to_numpy(), 0.0))


if __name__ == "__main__":
    unittest.main()
