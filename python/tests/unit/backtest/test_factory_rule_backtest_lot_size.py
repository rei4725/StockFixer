"""戦略ファクトリーとルールのバックテストが市場の単元で約定することのテスト（#978）。"""

from __future__ import annotations

import unittest
import unittest.mock

import pandas as pd

from src.backtest import factory, rule_backtester
from src.backtest.backtester import Backtester
from src.backtest.types import FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "rsi_contrarian", "params": {}}


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


def _prices(price: float) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({"Open": [price] * 10, "Close": [price] * 10}, index=index)


class _BuyQtySpy:
    """Backtester.simulate_trading を包み、買いの株数を集める。"""

    def __init__(self) -> None:
        self.buy_qty: list[int] = []
        self.lot_sizes: list[int] = []
        self.num_trades: list[int] = []
        self._orig = Backtester.simulate_trading

    def __call__(self, backtester, df, signal, *args, **kwargs):
        result_df, metrics = self._orig(backtester, df, signal, *args, **kwargs)
        self.lot_sizes.append(backtester.execution.costs.lot_size)
        self.num_trades.append(int(metrics.get("num_trades", 0) or 0))
        if not result_df.empty:
            buys = result_df[result_df["action"] == "buy"]
            self.buy_qty.extend(int(q) for q in buys["qty"])
        return result_df, metrics

    def patch(self):
        return unittest.mock.patch.object(
            Backtester, "simulate_trading", autospec=True, side_effect=self
        )


def _baseline_buy_qty(price: float) -> int:
    """市場を渡さない（従来の）Backtester で買う株数。"""
    backtester = Backtester(
        model_manager=None,
        signal_generator=None,
        data_loader=None,
        start_date=None,
        end_date=None,
        market="",
        symbol="",
        initial_cash=1_000_000,
        fee_rate=0.001,
        slippage=0.001,
        stop_loss_pct=0.07,
        take_profit_pct=None,
        position_sizing="full",
    )
    df = _prices(price)
    result_df, _ = backtester.simulate_trading(df, _StubRule().generate_signal(df))
    return int(result_df[result_df["action"] == "buy"]["qty"].iloc[0])


class TestFactoryLotSize(unittest.TestCase):
    def _evaluate(self, market: str, data: dict[str, pd.DataFrame]):
        spy = _BuyQtySpy()
        hypothesis = FactoryHypothesis(rule_spec=_SPEC, market=market)
        with unittest.mock.patch.object(factory, "build_rule", return_value=_StubRule()):
            with spy.patch():
                evaluation = factory.evaluate_hypothesis(hypothesis, data, windows=[])
        return spy, evaluation

    def test_jp_buys_in_multiples_of_100_and_skips_unaffordable_symbol(self):
        # 3,000円は 300 株（1単元100）、15,000円は1単元 150万円で 100万円の予算に届かない
        data = {"CHEAP": _prices(3_000.0), "PRICEY": _prices(15_000.0)}
        spy, _ = self._evaluate("jp", data)

        self.assertEqual(spy.lot_sizes, [100, 100])
        self.assertEqual(spy.buy_qty, [300])
        self.assertTrue(all(q % 100 == 0 for q in spy.buy_qty))
        # 銘柄の順（CHEAP → PRICEY）に約定数を見る
        self.assertGreater(spy.num_trades[0], 0)
        self.assertEqual(spy.num_trades[1], 0)

    def test_us_buy_qty_is_unchanged(self):
        spy, _ = self._evaluate("us", {"AAA": _prices(150.0)})

        self.assertEqual(spy.lot_sizes, [1])
        self.assertEqual(spy.buy_qty, [_baseline_buy_qty(150.0)])


class TestRuleBacktestLotSize(unittest.TestCase):
    def _run(self, market: str, price: float) -> _BuyQtySpy:
        spy = _BuyQtySpy()
        with unittest.mock.patch.object(
            rule_backtester,
            "get_backtest_data_port",
            return_value=_StubDataPort(_prices(price)),
        ):
            with spy.patch():
                rule_backtester.run_rule_backtest(
                    market=market,
                    symbol="7203",
                    rules=[_StubRule()],  # type: ignore[list-item]
                    start="2024-01-01",
                    end="2024-01-10",
                )
        return spy

    def test_jp_rule_backtest_uses_lot_100(self):
        spy = self._run("jp", 3_000.0)

        self.assertEqual(spy.lot_sizes, [100])
        self.assertEqual(spy.buy_qty, [300])

    def test_us_rule_backtest_buy_qty_is_unchanged(self):
        spy = self._run("us", 150.0)

        self.assertEqual(spy.lot_sizes, [1])
        self.assertEqual(spy.buy_qty, [_baseline_buy_qty(150.0)])


if __name__ == "__main__":
    unittest.main()
