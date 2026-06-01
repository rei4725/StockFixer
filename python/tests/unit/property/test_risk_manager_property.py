"""
Property-based tests for risk calculation functions.

Kelly criterion, drawdown scale, and position sizing constraints
are verified against algebraic invariants rather than example-based assertions.
"""

from unittest.mock import MagicMock, patch

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from src.trading.risk_manager import MAX_POSITION_RATE, RiskManager, compute_dd_capital_scale

# ---------------------------------------------------------------------------
# compute_dd_capital_scale
# ---------------------------------------------------------------------------

_dd_ratio = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_max_dd = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)


class TestComputeDdCapitalScaleProperty:
    @given(dd_ratio=_dd_ratio, max_dd=_max_dd)
    def test_scale_always_in_bounds(self, dd_ratio: float, max_dd: float) -> None:
        """任意の入力で scale は常に [0.25, 1.0] の範囲に収まる"""
        scale = compute_dd_capital_scale(dd_ratio, max_dd)
        assert 0.25 <= scale <= 1.0

    @given(
        dd_ratio1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        dd_ratio2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        max_dd=_max_dd,
    )
    def test_scale_monotonically_nonincreasing(
        self, dd_ratio1: float, dd_ratio2: float, max_dd: float
    ) -> None:
        """DD ratio が大きいほど scale は小さい（単調非増加）"""
        assume(dd_ratio1 <= dd_ratio2)
        scale1 = compute_dd_capital_scale(dd_ratio1, max_dd)
        scale2 = compute_dd_capital_scale(dd_ratio2, max_dd)
        assert scale1 >= scale2

    @given(
        dd_ratio=st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        max_dd=_max_dd,
    )
    def test_nonpositive_dd_ratio_returns_full_scale(self, dd_ratio: float, max_dd: float) -> None:
        """dd_ratio <= 0 なら scale = 1.0（縮小なし）"""
        scale = compute_dd_capital_scale(dd_ratio, max_dd)
        assert scale == 1.0

    @given(
        dd_ratio=_dd_ratio,
        max_dd=st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    def test_nonpositive_max_dd_returns_full_scale(self, dd_ratio: float, max_dd: float) -> None:
        """max_dd <= 0 なら scale = 1.0（縮小なし）"""
        scale = compute_dd_capital_scale(dd_ratio, max_dd)
        assert scale == 1.0


# ---------------------------------------------------------------------------
# calc_position_size
# ---------------------------------------------------------------------------

_win_rate = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_pct = st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False)
_balance = st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False)
_price = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)


def _make_risk(balance: float) -> RiskManager:
    broker = MagicMock()
    broker.get_balance.return_value = balance
    broker.get_positions.return_value = []
    broker.broker_name = "paper"
    risk = RiskManager(broker)
    risk.get_current_dd_ratio = MagicMock(return_value=0.0)
    return risk


_VIX_PATCH = "src.trading.risk_manager.fetch_latest_vix"


class TestPositionSizeProperty:
    @given(
        win_rate=_win_rate,
        avg_win=_pct,
        avg_loss=_pct,
        balance=_balance,
        price=_price,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_position_size_never_negative(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        balance: float,
        price: float,
    ) -> None:
        """任意の勝率・損益比・価格でポジションサイズは非負"""
        with patch(_VIX_PATCH, return_value=None):
            risk = _make_risk(balance)
            qty = risk.calc_position_size(
                "TEST", price=price, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss
            )
        assert qty >= 0

    @given(
        win_rate=_win_rate,
        avg_win=_pct,
        avg_loss=_pct,
        balance=_balance,
        price=_price,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_position_cost_within_max_position_rate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        balance: float,
        price: float,
    ) -> None:
        """発注コストが MAX_POSITION_RATE × 口座残高を超えない"""
        with patch(_VIX_PATCH, return_value=None):
            risk = _make_risk(balance)
            qty = risk.calc_position_size(
                "TEST", price=price, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss
            )
        assert qty * price <= balance * MAX_POSITION_RATE + 1e-6

    @given(
        win_rate=_win_rate,
        avg_win=_pct,
        avg_loss=_pct,
        balance=_balance,
        price=_price,
    )
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_position_size_multiple_of_lot(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        balance: float,
        price: float,
    ) -> None:
        """発注株数は 100 株単位（ロット制約）"""
        with patch(_VIX_PATCH, return_value=None):
            risk = _make_risk(balance)
            qty = risk.calc_position_size(
                "TEST", price=price, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss
            )
        assert qty % 100 == 0
