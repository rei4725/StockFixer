"""KabuBroker.get_positions の null の欄の扱い（#1010）"""

import logging
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import src.trading.brokers.kabu.kabu_client  # noqa: F401
from src.trading.position_valuation import positions_market_value

_GET = "src.trading.brokers.kabu.kabu_client.httpx.get"


def _broker():
    from src.trading.brokers.kabu.kabu_client import KabuBroker

    broker = KabuBroker(api_password="test_password")
    broker._token = "cached_token"
    broker._token_expires_at = datetime.now() + timedelta(hours=1)
    return broker


def _positions(rows):
    resp = MagicMock()
    resp.json.return_value = rows
    resp.raise_for_status = MagicMock()
    with patch(_GET, return_value=resp):
        return _broker().get_positions()


def test_null_current_price_becomes_nan_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        pos = _positions(
            [{"Symbol": "7203", "LeavesQty": 100, "Price": 1050.0, "CurrentPrice": None}]
        )[0]
    assert pos["current_price"] is not None
    assert math.isnan(pos["current_price"])
    assert pos["qty"] == 100
    assert pos["avg_price"] == 1050.0
    assert "7203" in caplog.text


def test_null_current_price_filled_from_valuation():
    pos = _positions(
        [
            {
                "Symbol": "7203",
                "LeavesQty": 100,
                "Price": 1050.0,
                "CurrentPrice": None,
                "Valuation": 110000.0,
            }
        ]
    )[0]
    assert pos["current_price"] == 1100.0


def test_null_price_and_qty_do_not_return_none(caplog):
    with caplog.at_level(logging.WARNING):
        pos = _positions(
            [{"Symbol": "9984", "LeavesQty": None, "Price": None, "CurrentPrice": None}]
        )[0]
    assert pos["qty"] == 0
    assert math.isnan(pos["avg_price"])
    assert math.isnan(pos["current_price"])
    assert "9984" in caplog.text


def test_total_equity_value_does_not_raise_on_null_row():
    rows = [
        {"Symbol": "7203", "LeavesQty": 100, "Price": 1050.0, "CurrentPrice": None},
        {"Symbol": "9984", "LeavesQty": 200, "Price": 5000.0, "CurrentPrice": 5200.0},
    ]
    value = positions_market_value(_positions(rows), logging.getLogger(__name__))
    assert value == 200 * 5200.0


def test_numbers_unchanged_without_null():
    pos = _positions(
        [{"Symbol": "9984", "LeavesQty": 200, "Price": 5000.0, "CurrentPrice": 5200.0}]
    )[0]
    assert pos == {
        "symbol": "9984",
        "qty": 200,
        "hold_qty": 0,
        "avg_price": 5000.0,
        "current_price": 5200.0,
    }
