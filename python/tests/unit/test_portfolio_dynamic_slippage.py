"""ポートフォリオのバックテストで動的スリッページ（市場インパクト）を数える（#1299）。

売りと買い（戦略と等分の基準の両方）に、約定の日より前の足までの 20 本の平均出来高を
volume として渡す。平均出来高に約定の日の出来高を含めない（backtester.py と同じ、PR 783）。
"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.slippage import make_slippage_fn

_SYM = "jp_TEST"
_PRICE = 1000.0
_THIN_VOLUME = 5000.0
_CASH = 1_000_000.0
_COSTS = TradingCosts(fee_rate=0.001, slippage_rate=0.0005, lot_size=100)


def _matrices(n_days=30, fill_day_volume=None):
    dates = pd.date_range("2026-01-05", periods=n_days, freq="B")
    score = pd.DataFrame({_SYM: [0.1] * n_days}, index=dates)
    close = pd.DataFrame({_SYM: [_PRICE] * n_days}, index=dates)
    volume = pd.DataFrame({_SYM: [_THIN_VOLUME] * n_days}, index=dates)
    if fill_day_volume is not None:
        volume.iloc[25, 0] = fill_day_volume
    return dates, score, close, volume


def _run(execution, volume_matrix):
    from src.backtest.portfolio import _simulate_portfolio

    dates, score, close, _ = _matrices()
    equity_df, _ = _simulate_portfolio(
        score,
        close,
        [dates[24]],
        top_n=1,
        initial_cash=_CASH,
        execution=execution,
        max_sector_positions=0,
        open_matrix=close,
        volume_matrix=volume_matrix,
    )
    # 約定の日（dates[25]）の評価額
    return equity_df.iloc[25]


def _expected_value(execution, avg_volume):
    qty = execution.max_affordable_qty(_CASH, _PRICE, avg_volume)
    return round(_CASH - execution.buy_cost(qty, _PRICE, avg_volume) + qty * _PRICE, 2)


@patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector", return_value="Tech")
class TestPortfolioDynamicSlippage(unittest.TestCase):
    def test_thin_symbol_buy_pays_market_impact(self, _sector):
        """平均出来高の小さい銘柄の買いで、現金の減りが上乗せの分だけ多い"""
        dynamic = ExecutionModel(_COSTS, make_slippage_fn())
        flat = ExecutionModel(_COSTS)
        _, _, _, volume = _matrices()

        row_dyn = _run(dynamic, volume)
        row_flat = _run(flat, volume)

        self.assertEqual(row_dyn["portfolio_value"], _expected_value(dynamic, _THIN_VOLUME))
        self.assertEqual(row_flat["portfolio_value"], _expected_value(flat, 0.0))
        # 900 株・平均出来高 5000 株の上乗せは 0.10 * sqrt(0.18) ≒ 4.2%
        impact = 0.10 * (900 / _THIN_VOLUME) ** 0.5 * 900 * _PRICE
        self.assertAlmostEqual(
            row_flat["portfolio_value"] - row_dyn["portfolio_value"], impact, delta=1.0
        )

    def test_equal_weight_leg_pays_market_impact(self, _sector):
        """等分の基準も同じ上乗せを払う"""
        dynamic = ExecutionModel(_COSTS, make_slippage_fn())
        _, _, _, volume = _matrices()

        row = _run(dynamic, volume)

        self.assertEqual(row["equal_weight_value"], _expected_value(dynamic, _THIN_VOLUME))

    def test_fill_day_volume_spike_does_not_change_cost(self, _sector):
        """約定の日の出来高が膨らんでも、その日の売買のコストは変わらない"""
        dynamic = ExecutionModel(_COSTS, make_slippage_fn())
        _, _, _, volume = _matrices()
        _, _, _, spiked = _matrices(fill_day_volume=50_000_000.0)

        row = _run(dynamic, volume)
        row_spiked = _run(dynamic, spiked)

        self.assertEqual(row_spiked["portfolio_value"], row["portfolio_value"])
        self.assertEqual(row_spiked["equal_weight_value"], row["equal_weight_value"])

    def test_without_volume_matrix_is_flat(self, _sector):
        """出来高の表が無いときはフラットの率だけ（今までと同じ）"""
        dynamic = ExecutionModel(_COSTS, make_slippage_fn())
        flat = ExecutionModel(_COSTS)

        self.assertEqual(
            _run(dynamic, None)["portfolio_value"], _run(flat, None)["portfolio_value"]
        )


class TestBuildExecutionDynamicSlippage(unittest.TestCase):
    def test_dynamic_slippage_sets_slippage_fn(self):
        from src.backtest.portfolio import _build_execution, _build_market_executions

        self.assertIsNotNone(_build_execution("jp", 0.001, None, True).slippage_fn)
        self.assertIsNone(_build_execution("jp", 0.001, None, False).slippage_fn)
        on = _build_market_executions(None, 0.001, None, True)
        off = _build_market_executions(None, 0.001, None, False)
        assert on is not None and off is not None
        self.assertTrue(all(e.slippage_fn is not None for e in on.values()))
        self.assertTrue(all(e.slippage_fn is None for e in off.values()))

    def test_default_is_dynamic(self):
        from src.backtest.portfolio import _build_execution

        self.assertIsNotNone(_build_execution("us", 0.001, None).slippage_fn)


@patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector", return_value="Tech")
class TestRunPortfolioBacktestDynamicSlippage(unittest.TestCase):
    def _run(self, **kwargs):
        from src.backtest.portfolio import run_portfolio_backtest

        _, score, close, volume = _matrices()
        with (
            patch("src.utils.db.stock_features.get_all_symbols", return_value=[("jp", "TEST")]),
            patch(
                "src.backtest.portfolio._build_signal_matrix",
                return_value=(score, close, close, volume),
            ),
            patch("src.backtest.portfolio._simulate_portfolio") as sim,
            patch("src.backtest.portfolio._compute_portfolio_metrics", return_value={}),
            patch("src.backtest.portfolio._attach_regime_metrics", return_value=(None, {})),
        ):
            sim.return_value = (pd.DataFrame(), [])
            run_portfolio_backtest(market="jp", **kwargs)
        return sim.call_args.kwargs

    def test_passes_volume_and_dynamic_execution(self, _sector):
        kwargs = self._run()
        self.assertIsNotNone(kwargs["execution"].slippage_fn)
        self.assertIsNotNone(kwargs["volume_matrix"])

    def test_dynamic_slippage_false_uses_flat(self, _sector):
        kwargs = self._run(dynamic_slippage=False)
        self.assertIsNone(kwargs["execution"].slippage_fn)


if __name__ == "__main__":
    unittest.main()
