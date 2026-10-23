"""ユニットテスト: ShadowBroker と make_order_broker"""

import json
from datetime import date
from unittest.mock import MagicMock

import pytest

import config.settings as cfg
from src.trading.brokers.account import factory
from src.trading.brokers.account.gateway import AccountGatewayBroker, AccountGatewayUnavailable
from src.trading.brokers.account.shadow import ShadowBroker
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.brokers.paper.paper_broker import PaperBroker

DAY = date(2026, 10, 26)


def _primary():
    p = MagicMock(spec=PaperBroker)
    p.broker_name = "paper"
    p.send_order.return_value = {"order_id": "abc", "status": "pending", "message": "ok"}
    p.get_positions.return_value = [{"symbol": "7203", "qty": 100}]
    p.settle_pending_orders.return_value = ["x"]
    return p


def _gateway():
    g = MagicMock(spec=AccountGatewayBroker)
    g.send_order.return_value = {"order_id": "1", "status": "pending", "message": "accepted"}
    return g


def _lines(tmp_path):
    text = (tmp_path / "2026-10-26.jsonl").read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines()]


def _shadow(tmp_path, primary, gateway, job="auto"):
    return ShadowBroker(primary, gateway, job=job, log_dir=str(tmp_path), today=lambda: DAY)


def test_buy_goes_to_both_and_returns_paper_result(tmp_path):
    p, g = _primary(), _gateway()
    s = _shadow(tmp_path, p, g)
    assert s.send_order("7203.T", OrderSide.BUY, 100) == {
        "order_id": "abc",
        "status": "pending",
        "message": "ok",
    }
    g.send_order.assert_called_once_with(
        "7203.T", OrderSide.BUY, 100, price=0.0, order_type=OrderType.MARKET
    )
    (line,) = _lines(tmp_path)
    assert line["job"] == "auto" and line["symbol"] == "7203" and line["side"] == "buy"
    assert line["paper"]["order_id"] == "abc" and line["account"]["status"] == "pending"


def test_shadow_failure_never_affects_paper(tmp_path):
    p, g = _primary(), _gateway()
    g.send_order.side_effect = AccountGatewayUnavailable("down")
    result = _shadow(tmp_path, p, g).send_order("7203", OrderSide.SELL, 100)
    assert result["order_id"] == "abc"
    (line,) = _lines(tmp_path)
    assert line["account"]["status"] == "error" and "down" in line["account"]["message"]


def test_unexpected_exception_is_swallowed_too(tmp_path):
    p, g = _primary(), _gateway()
    g.send_order.side_effect = RuntimeError("bug")
    assert _shadow(tmp_path, p, g).send_order("7203", OrderSide.BUY, 100)["order_id"] == "abc"


def test_short_is_not_forwarded(tmp_path):
    p, g = _primary(), _gateway()
    _shadow(tmp_path, p, g).send_order("7203", OrderSide.SHORT, 100)
    g.send_order.assert_not_called()
    assert _lines(tmp_path)[0]["account"]["status"] == "not_forwarded"


def test_paper_failure_is_not_forwarded(tmp_path):
    p, g = _primary(), _gateway()
    p.send_order.side_effect = RuntimeError("db down")
    with pytest.raises(RuntimeError):
        _shadow(tmp_path, p, g).send_order("7203", OrderSide.BUY, 100)
    g.send_order.assert_not_called()


def test_reads_and_paper_only_methods_go_to_primary(tmp_path):
    p, g = _primary(), _gateway()
    s = _shadow(tmp_path, p, g)
    assert s.broker_name == "paper"
    assert s.get_positions() == [{"symbol": "7203", "qty": 100}]
    assert s.settle_pending_orders() == ["x"]
    g.get_positions.assert_not_called()


def test_log_write_failure_does_not_raise(tmp_path):
    p, g = _primary(), _gateway()
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    s = ShadowBroker(p, g, job="auto", log_dir=str(blocker / "sub"), today=lambda: DAY)
    assert s.send_order("7203", OrderSide.BUY, 100)["order_id"] == "abc"


@pytest.fixture
def _cfg(monkeypatch):
    monkeypatch.setattr(cfg, "ACCOUNT_SHADOW", True)
    monkeypatch.setattr(cfg, "ACCOUNT_GATEWAY_URL", "http://account:8000")
    monkeypatch.setattr(cfg, "ACCOUNT_GATEWAY_TOKEN", "tok")
    return monkeypatch


def test_factory_returns_paper_when_shadow_off(_cfg):
    _cfg.setattr(cfg, "ACCOUNT_SHADOW", False)
    b = factory.make_order_broker(
        job="auto", market_data_port=MagicMock(), trade_diff_sink=MagicMock()
    )
    assert isinstance(b, PaperBroker)


def test_factory_falls_back_to_paper_without_url(_cfg):
    _cfg.setattr(cfg, "ACCOUNT_GATEWAY_URL", "")
    b = factory.make_order_broker(
        job="auto", market_data_port=MagicMock(), trade_diff_sink=MagicMock()
    )
    assert isinstance(b, PaperBroker)


def test_factory_returns_shadow_when_configured(_cfg):
    b = factory.make_order_broker(
        job="rule", market_data_port=MagicMock(), trade_diff_sink=MagicMock()
    )
    assert isinstance(b, ShadowBroker) and isinstance(b, BrokerBase)
    assert b.broker_name == "paper"
