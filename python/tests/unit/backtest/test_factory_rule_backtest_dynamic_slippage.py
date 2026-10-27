"""戦略ファクトリーとルールのバックテストが動的スリッページを数えることのテスト（#1244）。"""

from __future__ import annotations

import unittest
import unittest.mock

import pandas as pd

from src.backtest import factory, rule_backtester
from src.backtest.backtester import Backtester
from src.backtest.types import FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "rsi_contrarian", "params": {}}
_PRICE = 3_000.0


class _StubRule:
    """初日に買い、最終日の前に売るシグナルを返すルール。"""

    name = "stub"
    description = "stub"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index)
        signal.iloc[0] = 1
        signal.iloc[-2] = -1
        return signal


class _StubDataPort:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def download(self, ticker, start, end):
        return self._df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def _prices(with_volume: bool) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    data = {"Open": [_PRICE] * 10, "Close": [_PRICE] * 10}
    if with_volume:
        data["Volume"] = [10_000.0] * 10
    return pd.DataFrame(data, index=index)


class _Spy:
    """Backtester.simulate_trading を包み、slippage_fn と final_cash を集める。"""

    def __init__(self) -> None:
        self.slippage_fns: list = []
        self.final_cash: list[float] = []
        self._orig = Backtester.simulate_trading

    def __call__(self, backtester, df, signal, *args, **kwargs):
        result_df, metrics = self._orig(backtester, df, signal, *args, **kwargs)
        self.slippage_fns.append(backtester.execution.slippage_fn)
        self.final_cash.append(float(metrics["final_cash"]))
        return result_df, metrics

    def patch(self):
        return unittest.mock.patch.object(
            Backtester, "simulate_trading", autospec=True, side_effect=self
        )


def _flat_final_cash(df: pd.DataFrame) -> float:
    """slippage_fn を渡さない（直す前の）Backtester の final_cash。"""
    backtester = Backtester(
        model_manager=None,
        signal_generator=None,
        data_loader=None,
        start_date=None,
        end_date=None,
        market="jp",
        symbol="",
        initial_cash=1_000_000,
        fee_rate=0.001,
        slippage=0.001,
        stop_loss_pct=0.07,
        take_profit_pct=None,
        position_sizing="full",
    )
    _, metrics = backtester.simulate_trading(df, _StubRule().generate_signal(df))
    return float(metrics["final_cash"])


class TestRuleBacktestDynamicSlippage(unittest.TestCase):
    def _run(self, df: pd.DataFrame, **kwargs) -> _Spy:
        spy = _Spy()
        with unittest.mock.patch.object(
            rule_backtester, "get_backtest_data_port", return_value=_StubDataPort(df)
        ):
            with spy.patch():
                rule_backtester.run_rule_backtest(
                    market="jp",
                    symbol="7203",
                    rules=[_StubRule()],  # type: ignore[list-item]
                    start="2024-01-01",
                    end="2024-01-10",
                    **kwargs,
                )
        return spy

    def test_backtester_has_slippage_fn(self):
        spy = self._run(_prices(with_volume=True))

        self.assertEqual(len(spy.slippage_fns), 1)
        self.assertIsNotNone(spy.slippage_fns[0])

    def test_final_cash_is_lower_with_volume(self):
        df = _prices(with_volume=True)
        spy = self._run(df)

        self.assertLess(spy.final_cash[0], _flat_final_cash(df))

    def test_final_cash_is_unchanged_without_volume(self):
        df = _prices(with_volume=False)
        spy = self._run(df)

        self.assertAlmostEqual(spy.final_cash[0], _flat_final_cash(df))

    def test_dynamic_slippage_can_be_disabled(self):
        df = _prices(with_volume=True)
        spy = self._run(df, dynamic_slippage=False)

        self.assertIsNone(spy.slippage_fns[0])
        self.assertAlmostEqual(spy.final_cash[0], _flat_final_cash(df))


class TestFactoryDynamicSlippage(unittest.TestCase):
    def _evaluate(self, df: pd.DataFrame, **kwargs) -> _Spy:
        spy = _Spy()
        hypothesis = FactoryHypothesis(rule_spec=_SPEC, market="jp")
        with unittest.mock.patch.object(factory, "build_rule", return_value=_StubRule()):
            with spy.patch():
                factory.evaluate_hypothesis(hypothesis, {"AAA": df}, windows=[], **kwargs)
        return spy

    def test_backtester_has_slippage_fn(self):
        spy = self._evaluate(_prices(with_volume=True))

        self.assertEqual(len(spy.slippage_fns), 1)
        self.assertIsNotNone(spy.slippage_fns[0])

    def test_final_cash_is_lower_with_volume(self):
        df = _prices(with_volume=True)
        spy = self._evaluate(df)

        self.assertLess(spy.final_cash[0], _flat_final_cash(df))

    def test_final_cash_is_unchanged_without_volume(self):
        df = _prices(with_volume=False)
        spy = self._evaluate(df)

        self.assertAlmostEqual(spy.final_cash[0], _flat_final_cash(df))

    def test_dynamic_slippage_can_be_disabled(self):
        df = _prices(with_volume=True)
        spy = self._evaluate(df, dynamic_slippage=False)

        self.assertIsNone(spy.slippage_fns[0])
        self.assertAlmostEqual(spy.final_cash[0], _flat_final_cash(df))


if __name__ == "__main__":
    unittest.main()
