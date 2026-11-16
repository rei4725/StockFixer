"""domain の plan_limit_order と limit_order_fills の直接のテスト（#1920）。

しきい値は引数で渡し、config.settings は読まない。
"""

from __future__ import annotations

import pytest

from src.domain.limit_order_rules import LimitOrderPlan, limit_order_fills, plan_limit_order

_VOLUME_THRESHOLD = 100_000.0
_SPREAD_THRESHOLD = 0.03
_PRICE_BUFFER = 0.005


def _plan(
    *,
    market: str = "jp",
    is_buy_side: bool = True,
    current_price: float = 1_000.0,
    avg_volume: float | None = 1_000_000.0,
    spread_proxy: float | None = 0.01,
    price_buffer: float = _PRICE_BUFFER,
) -> LimitOrderPlan:
    return plan_limit_order(
        market=market,
        is_buy_side=is_buy_side,
        current_price=current_price,
        avg_volume=avg_volume,
        spread_proxy=spread_proxy,
        volume_threshold=_VOLUME_THRESHOLD,
        spread_threshold=_SPREAD_THRESHOLD,
        price_buffer=price_buffer,
    )


class TestPlanLimitOrder:
    def test_liquid_is_market_order(self) -> None:
        assert _plan() == LimitOrderPlan(False, 0.0, (), "open")

    def test_missing_liquidity_data_is_market_order(self) -> None:
        assert _plan(avg_volume=None, spread_proxy=None) == LimitOrderPlan(False, 0.0, (), "open")

    def test_exact_thresholds_are_market_order(self) -> None:
        # 出来高はしきい値未満、値幅はしきい値超えで初めて指値になる
        plan = _plan(avg_volume=_VOLUME_THRESHOLD, spread_proxy=_SPREAD_THRESHOLD)
        assert plan.is_limit is False

    def test_low_volume_only_is_close_limit(self) -> None:
        plan = _plan(avg_volume=50_000.0, spread_proxy=None)
        assert plan.is_limit is True
        assert plan.session == "close"
        assert plan.reasons == ("low_volume=50000",)
        # 1,000 円 * 0.5% = 5 円上、1円単位で切り下げても 1,005 円
        assert plan.limit_price == 1_005.0

    def test_wide_range_only_is_open_limit(self) -> None:
        plan = _plan(is_buy_side=False, avg_volume=1_000_000.0, spread_proxy=0.05)
        assert plan.is_limit is True
        assert plan.session == "open"
        assert plan.reasons == ("wide_range=5.000%",)
        assert plan.limit_price == 995.0

    def test_both_is_close_limit(self) -> None:
        plan = _plan(avg_volume=50_000.0, spread_proxy=0.05)
        assert plan.is_limit is True
        assert plan.session == "close"
        assert plan.reasons == ("low_volume=50000", "wide_range=5.000%")

    @pytest.mark.parametrize(("is_buy_side", "expected"), [(True, 4_005.0), (False, 3_995.0)])
    def test_buffer_is_at_least_one_tick(self, is_buy_side: bool, expected: float) -> None:
        # 4,000 円の帯の単位は5円。0.01% の余裕（0.4 円）は単位1つ分に広げる
        plan = _plan(
            is_buy_side=is_buy_side,
            current_price=4_000.0,
            avg_volume=50_000.0,
            price_buffer=0.0001,
        )
        assert plan.limit_price == expected

    def test_buy_rounds_down_and_sell_rounds_up_to_tick(self) -> None:
        # 3,500 円 * 0.3% = 10.5 円。買い 3,510.5 → 3,510、売り 3,489.5 → 3,490（5円単位）
        buy = _plan(current_price=3_500.0, avg_volume=50_000.0, price_buffer=0.003)
        sell = _plan(
            is_buy_side=False, current_price=3_500.0, avg_volume=50_000.0, price_buffer=0.003
        )
        assert buy.limit_price == 3_510.0
        assert sell.limit_price == 3_490.0

    def test_us_limit_rounds_to_cent(self) -> None:
        plan = _plan(market="us", current_price=123.45, avg_volume=50_000.0, price_buffer=0.001)
        # 123.45 + 0.12345 = 123.57345 → 切り下げて 123.57
        assert plan.limit_price == pytest.approx(123.57)

    @pytest.mark.parametrize("market", ["jp", "us"])
    def test_sell_limit_never_below_zero(self, market: str) -> None:
        # 余裕の幅（少なくとも単位1つ分）が現在値より大きくても、売りの指値は0を下回らない
        plan = _plan(
            market=market,
            is_buy_side=False,
            current_price=0.005,
            avg_volume=50_000.0,
            price_buffer=0.5,
        )
        assert plan.is_limit is True
        assert plan.limit_price >= 0.0
        assert plan.limit_price == 0.0


class TestLimitOrderFills:
    def test_market_order_always_fills(self) -> None:
        plan = LimitOrderPlan(False, 0.0, (), "open")
        assert limit_order_fills(plan, True, high=1.0, low=1.0, close=1.0) is True
        assert limit_order_fills(plan, False, high=1.0, low=1.0, close=1.0) is True

    def test_close_buy_judged_by_close_only(self) -> None:
        plan = LimitOrderPlan(True, 100.0, ("low_volume=1",), "close")
        assert limit_order_fills(plan, True, high=110.0, low=90.0, close=100.0) is True
        assert limit_order_fills(plan, True, high=110.0, low=90.0, close=99.0) is True
        # 安値は届いていても、終値が指値より上なら約定しない
        assert limit_order_fills(plan, True, high=110.0, low=90.0, close=100.5) is False

    def test_close_sell_judged_by_close_only(self) -> None:
        plan = LimitOrderPlan(True, 100.0, ("low_volume=1",), "close")
        assert limit_order_fills(plan, False, high=110.0, low=90.0, close=100.0) is True
        assert limit_order_fills(plan, False, high=110.0, low=90.0, close=101.0) is True
        # 高値は届いていても、終値が指値より下なら約定しない
        assert limit_order_fills(plan, False, high=110.0, low=90.0, close=99.5) is False

    def test_open_buy_judged_by_low(self) -> None:
        plan = LimitOrderPlan(True, 100.0, ("wide_range=5.000%",), "open")
        assert limit_order_fills(plan, True, high=110.0, low=100.0, close=105.0) is True
        assert limit_order_fills(plan, True, high=110.0, low=99.0, close=105.0) is True
        # 終値が指値より下でも、安値が届かなければ約定しない（終値は使わない）
        assert limit_order_fills(plan, True, high=110.0, low=100.5, close=100.5) is False

    def test_open_sell_judged_by_high(self) -> None:
        plan = LimitOrderPlan(True, 100.0, ("wide_range=5.000%",), "open")
        assert limit_order_fills(plan, False, high=100.0, low=90.0, close=95.0) is True
        assert limit_order_fills(plan, False, high=101.0, low=90.0, close=95.0) is True
        assert limit_order_fills(plan, False, high=99.5, low=90.0, close=99.5) is False
