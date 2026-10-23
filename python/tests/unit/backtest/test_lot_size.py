"""Unit Test: バックテストの株数を単元（jp=100株）の倍数に切り下げる（#965）

本番の発注は get_lot_size(market) の単元で丸める。バックテストが1株単位で
約定すると、本番では買えない株数・銘柄を成績に入れてしまう。
"""

import pytest

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.position_sizing import calc_position_details
from src.backtest.slippage import make_slippage_fn


def _model(market, fee_rate=0.0, slippage=0.0, slippage_fn=None):
    return ExecutionModel(
        TradingCosts.for_market(market, fee_rate=fee_rate, slippage=slippage),
        slippage_fn=slippage_fn,
    )


def _qty(market, cash, price, sizing, atr_value=None, fraction=0.5, fee_rate=0.0):
    return calc_position_details(
        cash=cash,
        price=price,
        position_sizing=sizing,
        position_fraction=fraction,
        atr_risk_pct=0.02,
        atr_multiplier=1.0,
        atr_min_fraction=0.1,
        atr_max_fraction=1.0,
        execution=_model(market, fee_rate=fee_rate),
        atr_value=atr_value,
    )["qty"]


class TestLotSizeOfCosts:
    def test_jp_lot_is_100(self):
        assert TradingCosts.for_market("jp").lot_size == 100

    def test_us_lot_is_1(self):
        assert TradingCosts.for_market("us").lot_size == 1

    def test_unknown_market_keeps_one_share(self):
        """市場の分からない呼び出し（market=""）は今の1株単位のままにする。"""
        assert TradingCosts.for_market("").lot_size == 1

    def test_default_is_one_share(self):
        assert TradingCosts().lot_size == 1


class TestMaxAffordableQtyLot:
    def test_jp_zero_when_budget_below_one_lot(self):
        # 150,000 / 2,000 = 75 株 < 100 株
        assert _model("jp").max_affordable_qty(150_000.0, 2000.0) == 0

    def test_jp_rounds_down_to_lot(self):
        # 450,000 / 2,000 = 225 株 → 200 株
        assert _model("jp").max_affordable_qty(450_000.0, 2000.0) == 200

    def test_jp_with_costs_never_overdraws(self):
        """#483: 単元で切り下げても買った後の現金が負にならない。"""
        m = _model("jp", fee_rate=0.001, slippage=0.001, slippage_fn=make_slippage_fn(alpha=0.10))
        cash = 400_000.0
        n = m.max_affordable_qty(cash, 2000.0, volume=50_000.0)
        assert n % 100 == 0
        assert n > 0
        assert m.buy_cost(n, 2000.0, 50_000.0) <= cash
        assert m.buy_cost(n + 100, 2000.0, 50_000.0) > cash

    def test_us_unchanged(self):
        assert _model("us").max_affordable_qty(150_000.0, 2000.0) == 75
        assert _model("us").max_affordable_qty(450_000.0, 2000.0) == 225


class TestCalcPositionDetailsLot:
    @pytest.mark.parametrize(
        "sizing,cash,atr_value",
        [("full", 150_000.0, None), ("fixed", 300_000.0, None), ("atr", 150_000.0, 50.0)],
    )
    def test_jp_zero_when_budget_below_one_lot(self, sizing, cash, atr_value):
        assert _qty("jp", cash, 2000.0, sizing, atr_value=atr_value) == 0

    @pytest.mark.parametrize(
        "sizing,cash,atr_value,expected",
        [
            ("full", 450_000.0, None, 200),
            ("fixed", 900_000.0, None, 200),
            ("atr", 1_000_000.0, 70.0, 200),
        ],
    )
    def test_jp_rounds_down_to_lot(self, sizing, cash, atr_value, expected):
        assert _qty("jp", cash, 2000.0, sizing, atr_value=atr_value) == expected

    @pytest.mark.parametrize(
        "sizing,cash,atr_value,expected",
        [
            ("full", 150_000.0, None, 75),
            ("full", 450_000.0, None, 225),
            ("fixed", 300_000.0, None, 75),
            ("fixed", 900_000.0, None, 225),
            ("atr", 150_000.0, 50.0, 60),
            ("atr", 1_000_000.0, 70.0, 285),
        ],
    )
    def test_us_unchanged(self, sizing, cash, atr_value, expected):
        assert _qty("us", cash, 2000.0, sizing, atr_value=atr_value) == expected

    @pytest.mark.parametrize("sizing", ["full", "fixed", "atr"])
    def test_jp_with_costs_never_overdraws(self, sizing):
        cash = 1_000_000.0
        m = _model("jp", fee_rate=0.001)
        qty = _qty("jp", cash, 2000.0, sizing, atr_value=70.0, fee_rate=0.001)
        assert qty % 100 == 0
        assert m.buy_cost(qty, 2000.0) <= cash
