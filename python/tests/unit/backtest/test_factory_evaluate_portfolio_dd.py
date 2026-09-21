"""evaluate_hypothesis がポートフォリオDDを算出することのテスト（ゲート指標の是正）。

有効銘柄フィルタを通った銘柄の equity 曲線だけを等金額で合成する。
除外銘柄の曲線が混ざると Sharpe と DD の母集団がずれるため、その回帰も張る。
"""

from __future__ import annotations

import math
import unittest
import unittest.mock

import pandas as pd

from src.backtest import factory
from src.backtest.types import FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "rsi_contrarian", "params": {}}
_INITIAL = 1_000_000.0
_DATES = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])


class _StubRule:
    def __init__(self, signal: pd.Series) -> None:
        self._signal = signal

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return self._signal


class _StubBacktester:
    """銘柄ごとに (metrics, equity_curve) を順に返す Backtester。"""

    def __init__(self, metrics_list: list[dict]) -> None:
        self._it = iter(metrics_list)

    def simulate_trading(self, df, signal, collect_equity: bool = False):
        return None, next(self._it)


def _metrics(num_trades: int, max_drawdown: float, equity: list[float]) -> dict:
    return {
        "num_trades": num_trades,
        "sharpe_ratio": 0.5,
        "sharpe_per_trade": 0.05,
        "win_rate": 0.6,
        "total_return": 0.1,
        "max_drawdown": max_drawdown,
        "equity_curve": pd.Series([v * _INITIAL for v in equity], index=_DATES),
    }


class TestEvaluateHypothesisPortfolioDrawdown(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"Close": range(4)}, index=_DATES)
        self.signal = pd.Series([1] * 4, index=_DATES)

    def _run(self, metrics_list: list[dict], symbols: list[str], min_trades_per_symbol: int):
        data = {s: self.df for s in symbols}
        hypothesis = FactoryHypothesis(rule_spec=_SPEC, market="jp")
        with unittest.mock.patch.object(
            factory, "build_rule", return_value=_StubRule(self.signal)
        ), unittest.mock.patch.object(
            factory, "_make_backtester", return_value=_StubBacktester(metrics_list)
        ):
            return factory.evaluate_hypothesis(
                hypothesis,
                data,
                windows=[],
                initial_cash=_INITIAL,
                min_trades_per_symbol=min_trades_per_symbol,
            )

    def test_portfolio_drawdown_averages_effective_symbol_curves(self):
        # BBB 単体は -50%、CCC は 0%。等金額ポートフォリオでは相殺されて 0%。
        result = self._run(
            [
                _metrics(8, -0.50, [1.0, 0.5, 1.0, 1.0]),
                _metrics(8, 0.0, [1.0, 1.5, 1.5, 1.5]),
            ],
            symbols=["BBB", "CCC"],
            min_trades_per_symbol=3,
        )

        self.assertAlmostEqual(result.portfolio_max_drawdown, 0.0)
        # 最悪銘柄DD は診断値として従来どおり残す
        self.assertAlmostEqual(result.max_drawdown, -0.50)

    def test_excluded_symbol_curve_is_not_mixed_in(self):
        # AAA は2取引で除外。その壊滅的な曲線は合成に混ぜない。
        result = self._run(
            [
                _metrics(2, -0.90, [1.0, 0.1, 0.1, 0.1]),
                _metrics(8, -0.50, [1.0, 0.5, 1.0, 1.0]),
                _metrics(8, 0.0, [1.0, 1.5, 1.5, 1.5]),
            ],
            symbols=["AAA", "BBB", "CCC"],
            min_trades_per_symbol=3,
        )

        self.assertEqual(result.n_effective_symbols, 2)
        self.assertAlmostEqual(result.portfolio_max_drawdown, 0.0)

    def test_portfolio_drawdown_is_nan_without_usable_curves(self):
        metrics = _metrics(8, -0.30, [1.0, 1.0, 1.0, 1.0])
        del metrics["equity_curve"]

        result = self._run([metrics], symbols=["BBB"], min_trades_per_symbol=3)

        self.assertTrue(math.isnan(result.portfolio_max_drawdown))


if __name__ == "__main__":
    unittest.main()
