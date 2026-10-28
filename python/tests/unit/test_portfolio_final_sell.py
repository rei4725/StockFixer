"""Issue #1332: 最後の日に残った保有を、その日の終値で sell_proceeds を通して売った現金で評価する。

単一銘柄のバックテストの final_sell（backtester.py）と同じく、期末の保有にも
売りの手数料・スリッページ・市場インパクトを数える。等分の基準も同じにする。
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.slippage import make_slippage_fn

_SYM = "jp_TEST"
_CASH = 1_000_000.0
_OPEN = 1000.0
_CLOSES = [1000.0, 1000.0, 1050.0, 1100.0, 1200.0]
_COSTS = TradingCosts(fee_rate=0.001, slippage_rate=0.0005, lot_size=100)
_ZERO_COSTS = TradingCosts(fee_rate=0.0, slippage_rate=0.0, lot_size=100)


def _run(execution, n_days=len(_CLOSES), closes=None, volume_matrix=None, rebalance_idx=0):
    from src.backtest.portfolio import _simulate_portfolio

    closes = closes if closes is not None else _CLOSES
    dates = pd.date_range("2026-01-05", periods=n_days, freq="B")
    score = pd.DataFrame({_SYM: [0.1] * n_days}, index=dates)
    close = pd.DataFrame({_SYM: closes}, index=dates)
    open_ = pd.DataFrame({_SYM: [_OPEN] * n_days}, index=dates)
    with patch(
        "src.backtest.portfolio.simulation._get_portfolio_symbol_sector", return_value="Tech"
    ):
        equity_df, _ = _simulate_portfolio(
            score,
            close,
            [dates[rebalance_idx]],
            top_n=1,
            initial_cash=_CASH,
            execution=execution,
            max_sector_positions=0,
            open_matrix=open_,
            volume_matrix=volume_matrix,
        )
    return equity_df


def _cash_after_buy(execution, volume=0.0):
    qty = execution.max_affordable_qty(_CASH, _OPEN, volume)
    return qty, _CASH - execution.buy_cost(qty, _OPEN, volume)


class TestPortfolioFinalSell:
    def test_last_day_value_is_cash_after_sell(self):
        """最後の日の評価は、終値で sell_proceeds を通して売った後の現金になる"""
        execution = ExecutionModel(_COSTS)
        qty, cash = _cash_after_buy(execution)

        equity = _run(execution)

        expected = round(cash + execution.sell_proceeds(qty, _CLOSES[-1]), 2)
        assert equity.iloc[-1]["portfolio_value"] == pytest.approx(expected)
        assert equity.iloc[-1]["equal_weight_value"] == pytest.approx(expected)

    def test_earlier_days_are_marked_at_close(self):
        """最後の日より前の日は、今までどおり終値の時価で評価する"""
        execution = ExecutionModel(_COSTS)
        qty, cash = _cash_after_buy(execution)

        equity = _run(execution)

        assert equity.iloc[0]["portfolio_value"] == pytest.approx(_CASH)
        for i in range(1, len(_CLOSES) - 1):
            expected = round(cash + qty * _CLOSES[i], 2)
            assert equity.iloc[i]["portfolio_value"] == pytest.approx(expected)
            assert equity.iloc[i]["equal_weight_value"] == pytest.approx(expected)

    def test_positive_cost_lowers_last_value_by_sell_cost(self):
        """コストが正なら、最後の日の評価が売りのコストの分だけ時価より小さい"""
        execution = ExecutionModel(_COSTS)
        qty, cash = _cash_after_buy(execution)

        equity = _run(execution)

        marked = cash + qty * _CLOSES[-1]
        sell_cost = qty * _CLOSES[-1] * (_COSTS.fee_rate + _COSTS.slippage_rate)
        assert sell_cost > 0
        assert equity.iloc[-1]["portfolio_value"] == pytest.approx(marked - sell_cost, abs=0.01)

    def test_zero_cost_keeps_last_value(self):
        """コストが 0 なら、最後の日の評価は終値の時価と同じ"""
        execution = ExecutionModel(_ZERO_COSTS)
        qty, cash = _cash_after_buy(execution)

        equity = _run(execution)

        marked = round(cash + qty * _CLOSES[-1], 2)
        assert equity.iloc[-1]["portfolio_value"] == pytest.approx(marked)
        assert equity.iloc[-1]["equal_weight_value"] == pytest.approx(marked)

    def test_no_holdings_last_value_is_cash(self):
        """最後の日に保有が無ければ、評価は現金のまま"""
        equity = _run(ExecutionModel(_COSTS), rebalance_idx=len(_CLOSES) - 1)

        assert equity.iloc[-1]["portfolio_value"] == pytest.approx(_CASH)
        assert equity.iloc[-1]["equal_weight_value"] == pytest.approx(_CASH)


class TestPortfolioFinalSellDynamicSlippage:
    _N = 30
    _THIN = 5000.0

    def _volume(self, last_day_volume=None):
        dates = pd.date_range("2026-01-05", periods=self._N, freq="B")
        volume = pd.DataFrame({_SYM: [self._THIN] * self._N}, index=dates)
        if last_day_volume is not None:
            volume.iloc[-1, 0] = last_day_volume
        return volume

    def _run(self, execution, volume):
        return _run(
            execution,
            n_days=self._N,
            closes=[_OPEN] * self._N,
            volume_matrix=volume,
            rebalance_idx=24,
        )

    def test_final_sell_pays_market_impact_with_prior_volume(self):
        """期末の売りにも、その日より前の平均出来高で市場インパクトを数える"""
        dynamic = ExecutionModel(_COSTS, make_slippage_fn())
        qty, cash = _cash_after_buy(dynamic, self._THIN)

        equity = self._run(dynamic, self._volume())

        expected = round(cash + dynamic.sell_proceeds(qty, _OPEN, self._THIN), 2)
        assert equity.iloc[-1]["portfolio_value"] == pytest.approx(expected)
        assert equity.iloc[-1]["equal_weight_value"] == pytest.approx(expected)
        flat_sell = ExecutionModel(_COSTS).sell_proceeds(qty, _OPEN)
        assert dynamic.sell_proceeds(qty, _OPEN, self._THIN) < flat_sell

    def test_last_day_volume_spike_does_not_change_final_sell(self):
        """最後の日の出来高が膨らんでも、期末の売りのコストは変わらない"""
        dynamic = ExecutionModel(_COSTS, make_slippage_fn())

        equity = self._run(dynamic, self._volume())
        spiked = self._run(dynamic, self._volume(last_day_volume=50_000_000.0))

        assert spiked.iloc[-1]["portfolio_value"] == equity.iloc[-1]["portfolio_value"]
        assert spiked.iloc[-1]["equal_weight_value"] == equity.iloc[-1]["equal_weight_value"]
