"""Unit Test: ExecutionModel / TradingCosts（Phase 1 / PR-1）

backtest BC に 21 箇所へ散っていた手数料・スリッページ演算を集約した型のテスト。
設計: docs/superpowers/specs/2026-09-21-execution-model-design.md
"""

from dataclasses import FrozenInstanceError

import pytest

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.slippage import make_slippage_fn


def _model(fee_rate=0.0, slippage_rate=0.0, slippage_fn=None):
    return ExecutionModel(
        TradingCosts(fee_rate=fee_rate, slippage_rate=slippage_rate),
        slippage_fn=slippage_fn,
    )


class TestEffectiveSlippage:
    """Backtester._get_slippage の分岐をそのまま型の側へ移した回帰ケース。"""

    def test_base_only_without_fn(self):
        assert _model(slippage_rate=0.001).effective_slippage(100, 50.0, 1000.0) == pytest.approx(
            0.001
        )

    def test_adds_dynamic_impact(self):
        # 動的項 = alpha(0.10) * sqrt(qty/vol) = 0.10 * sqrt(100/400) = 0.05
        m = _model(slippage_rate=0.001, slippage_fn=make_slippage_fn(alpha=0.10))
        assert m.effective_slippage(100, 50.0, 400.0) == pytest.approx(0.001 + 0.05)

    def test_base_only_when_no_volume(self):
        m = _model(slippage_rate=0.001, slippage_fn=make_slippage_fn(alpha=0.10))
        assert m.effective_slippage(100, 50.0, 0.0) == pytest.approx(0.001)

    def test_base_only_when_zero_qty(self):
        m = _model(slippage_rate=0.001, slippage_fn=make_slippage_fn(alpha=0.10))
        assert m.effective_slippage(0, 50.0, 1000.0) == pytest.approx(0.001)

    def test_dynamic_increases_with_participation(self):
        m = _model(slippage_fn=make_slippage_fn(alpha=0.10))
        small = m.effective_slippage(10, 50.0, 10_000.0)
        large = m.effective_slippage(1000, 50.0, 10_000.0)
        assert large > small > 0


class TestBuySellRoundTrip:
    def test_costless_round_trip_is_exact(self):
        m = _model()
        assert m.buy_cost(10, 100.0) == pytest.approx(1000.0)
        assert m.sell_proceeds(10, 100.0) == pytest.approx(1000.0)

    def test_costs_squeeze_both_sides(self):
        m = _model(fee_rate=0.001, slippage_rate=0.0005)
        gross = 10 * 100.0
        assert m.sell_proceeds(10, 100.0) < gross < m.buy_cost(10, 100.0)

    def test_buy_cost_matches_formula(self):
        m = _model(fee_rate=0.001, slippage_rate=0.0005)
        assert m.buy_cost(10, 100.0) == pytest.approx(10 * 100.0 * (1 + 0.001 + 0.0005))

    def test_sell_proceeds_matches_formula(self):
        m = _model(fee_rate=0.001, slippage_rate=0.0005)
        assert m.sell_proceeds(10, 100.0) == pytest.approx(10 * 100.0 * (1 - 0.001 - 0.0005))

    def test_sell_proceeds_accepts_fractional_shares(self):
        """longterm の部分利確は端株を売る（original_shares * current_hf）。"""
        m = _model(fee_rate=0.001)
        assert m.sell_proceeds(2.5, 100.0) == pytest.approx(2.5 * 100.0 * (1 - 0.001))

    def test_dynamic_slippage_reaches_both_sides(self):
        m = _model(slippage_fn=make_slippage_fn(alpha=0.10))
        # volume を与えると動的インパクトが乗り、無コストより不利になる
        assert m.buy_cost(100, 50.0, volume=400.0) > m.buy_cost(100, 50.0, volume=0.0)
        assert m.sell_proceeds(100, 50.0, volume=400.0) < m.sell_proceeds(100, 50.0, volume=0.0)


class TestUnitBuyCost:
    def test_matches_position_sizing_formula(self):
        """position_sizing.py:45 の unit_cost = price * (1 + fee + slippage)。"""
        m = _model(fee_rate=0.001, slippage_rate=0.0005)
        assert m.unit_buy_cost(100.0) == pytest.approx(100.0 * (1 + 0.001 + 0.0005))

    def test_is_buy_cost_of_one_share(self):
        m = _model(fee_rate=0.002)
        assert m.unit_buy_cost(250.0) == pytest.approx(m.buy_cost(1, 250.0))


class TestMaxAffordableQty:
    def test_never_overdraws_cash(self):
        """買った後に現金が負にならない。過剰買付（#483 の系譜）の再発防止。"""
        m = _model(fee_rate=0.001, slippage_rate=0.0005)
        cash = 10_000.0
        n = m.max_affordable_qty(cash, 99.0)
        assert m.buy_cost(n, 99.0) <= cash
        assert m.buy_cost(n + 1, 99.0) > cash

    def test_costless_case(self):
        m = _model()
        assert m.max_affordable_qty(1000.0, 100.0) == 10

    def test_returns_zero_when_unaffordable(self):
        m = _model(fee_rate=0.001)
        assert m.max_affordable_qty(50.0, 100.0) == 0

    def test_returns_zero_for_nonpositive_inputs(self):
        m = _model()
        assert m.max_affordable_qty(0.0, 100.0) == 0
        assert m.max_affordable_qty(1000.0, 0.0) == 0
        assert m.max_affordable_qty(-10.0, 100.0) == 0


class TestTradingCostsForMarket:
    def test_us_default(self):
        assert TradingCosts.for_market("us").slippage_rate == pytest.approx(0.0005)

    def test_jp_default(self):
        assert TradingCosts.for_market("jp").slippage_rate == pytest.approx(0.0010)

    def test_unknown_market_falls_back_to_us(self):
        assert TradingCosts.for_market("xx").slippage_rate == pytest.approx(0.0005)

    def test_explicit_slippage_overrides_default(self):
        assert TradingCosts.for_market("jp", slippage=0.002).slippage_rate == pytest.approx(0.002)

    def test_explicit_zero_slippage_is_respected(self):
        """0.0 は「未指定」ではない。None との区別が崩れると規律が消える。"""
        assert TradingCosts.for_market("jp", slippage=0.0).slippage_rate == pytest.approx(0.0)

    def test_fee_rate_default(self):
        assert TradingCosts.for_market("us").fee_rate == pytest.approx(0.001)

    def test_is_frozen(self):
        costs = TradingCosts(fee_rate=0.001)
        with pytest.raises(FrozenInstanceError):
            costs.fee_rate = 0.002  # type: ignore[misc]
