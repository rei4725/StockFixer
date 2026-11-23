"""ユニットテスト: 保有の数え方の部品 trading/position_valuation.py を直に確かめる（#2129）

小さな偽のブローカー（get_positions と get_short_positions だけ）と dict の行で、
返済の株数・落ちた空売りの見分け・空売りの銘柄コード・総資産の足し算の形を固める。
期待値はすべて手で数えた値である。
"""

import logging
import math
from typing import Any

import pytest

from src.trading.position_valuation import (
    broker_positions_value,
    count_short_positions,
    coverable_short_qty,
    is_failed_short_row,
    long_positions_unrealized_pnl,
    short_held_codes,
    short_positions_unrealized_pnl,
)

LOGGER = logging.getLogger(__name__)
NAN = float("nan")


class _FakeBroker:
    """get_positions と get_short_positions だけを持つ偽のブローカー。"""

    def __init__(
        self,
        positions: list[dict[str, Any]] | None = None,
        shorts: list[dict[str, Any]] | None = None,
    ) -> None:
        self._positions = positions or []
        self._shorts = shorts or []

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self._positions)

    def get_short_positions(self) -> list[dict[str, Any]]:
        return list(self._shorts)


# --- long_positions_unrealized_pnl ---


def test_long_pnl_sums_rows() -> None:
    rows = [
        {"symbol": "7203.T", "unrealized_pnl": 1500.0},
        {"symbol": "6758.T", "unrealized_pnl": -300.5},
    ]
    assert long_positions_unrealized_pnl(rows, LOGGER) == pytest.approx(1199.5)


def test_long_pnl_skips_nan_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        {"symbol": "7203.T", "unrealized_pnl": 1500.0},
        {"symbol": "6758.T", "unrealized_pnl": NAN},
        {"symbol": "9984.T", "unrealized_pnl": -300.5},
    ]
    with caplog.at_level(logging.WARNING, logger=__name__):
        total = long_positions_unrealized_pnl(rows, LOGGER)
    assert total == pytest.approx(1199.5)
    assert not math.isnan(total)
    assert len(caplog.records) == 1
    assert "6758.T" in caplog.records[0].getMessage()


def test_long_pnl_missing_key_counts_zero_and_empty_is_zero() -> None:
    assert long_positions_unrealized_pnl([{"symbol": "7203.T"}], LOGGER) == 0.0
    assert long_positions_unrealized_pnl([], LOGGER) == 0.0


# --- short_positions_unrealized_pnl ---


def test_short_pnl_sums_rows() -> None:
    rows = [
        {"symbol": "7203.T", "unrealized_pnl": -200.0},
        {"symbol": "6758.T", "unrealized_pnl": 50.25},
    ]
    assert short_positions_unrealized_pnl(rows, LOGGER) == pytest.approx(-149.75)


def test_short_pnl_skips_nan_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    rows = [
        {"symbol": "7203.T", "unrealized_pnl": -200.0},
        {"symbol": "6758.T", "unrealized_pnl": NAN},
        {"symbol": "9984.T", "unrealized_pnl": 50.25},
    ]
    with caplog.at_level(logging.WARNING, logger=__name__):
        total = short_positions_unrealized_pnl(rows, LOGGER)
    assert total == pytest.approx(-149.75)
    assert len(caplog.records) == 1
    assert "6758.T" in caplog.records[0].getMessage()


def test_short_pnl_all_nan_is_zero() -> None:
    rows = [{"symbol": "7203.T", "unrealized_pnl": NAN}]
    assert short_positions_unrealized_pnl(rows, LOGGER) == 0.0


# --- coverable_short_qty ---


@pytest.mark.parametrize(
    ("pos", "expected"),
    [
        ({"qty": 300, "pending_qty": 100}, 200),
        ({"qty": 500}, 500),
        ({"qty": 100, "pending_qty": 100}, 0),
        ({"qty": 100, "pending_qty": 300}, -200),
        ({"qty": None, "pending_qty": 100}, -100),
        ({"qty": 400, "pending_qty": None}, 400),
        ({"qty": None, "pending_qty": None}, 0),
    ],
)
def test_coverable_short_qty(pos: dict[str, Any], expected: int) -> None:
    assert coverable_short_qty(pos) == expected


# --- is_failed_short_row ---


def test_failed_short_row_all_three_conditions() -> None:
    assert is_failed_short_row({"qty": 100, "pending_qty": 100, "unrealized_pnl": NAN}) is True


def test_failed_short_row_pending_exceeds_qty() -> None:
    assert is_failed_short_row({"qty": 100, "pending_qty": 300, "unrealized_pnl": NAN}) is True


def test_failed_short_row_false_without_positive_qty() -> None:
    assert is_failed_short_row({"qty": 0, "pending_qty": 100, "unrealized_pnl": NAN}) is False
    assert is_failed_short_row({"qty": None, "pending_qty": 100, "unrealized_pnl": NAN}) is False


def test_failed_short_row_false_with_coverable_qty() -> None:
    assert is_failed_short_row({"qty": 100, "pending_qty": 50, "unrealized_pnl": NAN}) is False
    assert is_failed_short_row({"qty": 100, "pending_qty": None, "unrealized_pnl": NAN}) is False


def test_failed_short_row_false_without_nan_pnl() -> None:
    assert is_failed_short_row({"qty": 100, "pending_qty": 100, "unrealized_pnl": 0.0}) is False
    assert is_failed_short_row({"qty": 100, "pending_qty": 100, "unrealized_pnl": None}) is False
    assert is_failed_short_row({"qty": 100, "pending_qty": 100}) is False


# --- count_short_positions ---


def test_count_short_positions_counts_positive_qty_only() -> None:
    broker = _FakeBroker(
        shorts=[
            {"symbol": "7203.T", "qty": 100},
            {"symbol": "6758.T", "qty": 0},
            {"symbol": "9984", "qty": 200},
            {"symbol": "8306.T"},
        ]
    )
    assert count_short_positions(broker) == 2


def test_count_short_positions_empty() -> None:
    assert count_short_positions(_FakeBroker()) == 0


# --- short_held_codes ---


def test_short_held_codes_strips_dot_t() -> None:
    broker = _FakeBroker(
        shorts=[
            {"symbol": "7203.T", "qty": 100},
            {"symbol": "9984", "qty": 200},
            {"symbol": "6758.T", "qty": 0},
        ]
    )
    assert short_held_codes(broker) == {"7203", "9984"}


def test_short_held_codes_merges_same_code_and_empty() -> None:
    broker = _FakeBroker(shorts=[{"symbol": "7203.T", "qty": 100}, {"symbol": "7203", "qty": 50}])
    assert short_held_codes(broker) == {"7203"}
    assert short_held_codes(_FakeBroker()) == set()


# --- broker_positions_value ---


def test_broker_positions_value_adds_long_value_and_short_pnl() -> None:
    broker = _FakeBroker(
        positions=[
            {"symbol": "7203.T", "qty": 100, "current_price": 1500.0},
            {"symbol": "6758.T", "qty": 10, "current_price": 3000.0},
        ],
        shorts=[
            {"symbol": "9984.T", "qty": 100, "unrealized_pnl": -2000.0},
            {"symbol": "8306.T", "qty": 200, "unrealized_pnl": 500.0},
        ],
    )
    # 100*1500 + 10*3000 = 180000、-2000 + 500 = -1500
    assert broker_positions_value(broker, LOGGER) == pytest.approx(178500.0)


def test_broker_positions_value_skips_nan_rows() -> None:
    broker = _FakeBroker(
        positions=[
            {"symbol": "7203.T", "qty": 100, "current_price": 1500.0},
            {"symbol": "6758.T", "qty": 10, "current_price": NAN},
        ],
        shorts=[
            {"symbol": "9984.T", "qty": 100, "unrealized_pnl": -2000.0},
            {"symbol": "8306.T", "qty": 100, "unrealized_pnl": NAN},
        ],
    )
    assert broker_positions_value(broker, LOGGER) == pytest.approx(148000.0)


def test_broker_positions_value_empty_and_shorts_only() -> None:
    assert broker_positions_value(_FakeBroker(), LOGGER) == 0.0
    broker = _FakeBroker(shorts=[{"symbol": "9984.T", "qty": 100, "unrealized_pnl": -2000.0}])
    assert broker_positions_value(broker, LOGGER) == pytest.approx(-2000.0)
