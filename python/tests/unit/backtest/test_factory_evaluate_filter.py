"""evaluate_hypothesis が少取引銘柄を集計から除外することのテスト（#625）。"""

from __future__ import annotations

import unittest
import unittest.mock

import pandas as pd

from src.backtest import factory
from src.backtest.types import FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "rsi_contrarian", "params": {}}


class _StubRule:
    """常に全日買いシグナルを返すルール。"""

    def __init__(self, signal: pd.Series) -> None:
        self._signal = signal

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return self._signal


class _StubBacktester:
    """呼び出し順に既定のメトリクスを返す Backtester。"""

    def __init__(self, metrics_list: list[dict]) -> None:
        self._it = iter(metrics_list)

    def simulate_trading(self, df, signal):
        return None, next(self._it)


def _metrics(num_trades: int, sharpe: float) -> dict:
    return {
        "num_trades": num_trades,
        "sharpe_ratio": sharpe,
        "sharpe_per_trade": sharpe / 10,
        "win_rate": 0.9,
        "total_return": 0.1,
        "max_drawdown": -0.05,
    }


class TestEvaluateHypothesisFilter(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D")
        self.df = pd.DataFrame({"Close": range(10)}, index=index)
        self.signal = pd.Series([1] * 10, index=index)

    def _run(self, metrics_list: list[dict], min_trades_per_symbol: int):
        """スタブを差し込んだ状態で evaluate_hypothesis を1回実行する。"""
        data = {"AAA": self.df, "BBB": self.df}
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
                min_trades_per_symbol=min_trades_per_symbol,
            )

    def test_low_trade_symbol_is_excluded_from_average(self):
        result = self._run([_metrics(2, 25.0), _metrics(8, 0.5)], min_trades_per_symbol=3)

        self.assertEqual(result.n_symbols, 2)
        self.assertEqual(result.n_symbols_with_signal, 2)
        self.assertEqual(result.n_effective_symbols, 1)
        self.assertAlmostEqual(result.sharpe_ratio, 0.5)
        self.assertEqual(result.num_trades, 8)
        self.assertAlmostEqual(result.avg_trades_per_symbol, 5.0)

    def test_threshold_one_keeps_every_symbol(self):
        result = self._run([_metrics(2, 25.0), _metrics(8, 0.5)], min_trades_per_symbol=1)

        self.assertEqual(result.n_effective_symbols, 2)
        self.assertAlmostEqual(result.sharpe_ratio, 12.75)


if __name__ == "__main__":
    unittest.main()
