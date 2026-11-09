"""KabuBroker の取引パスワードの欠けの確かめのテスト（Issue 1009）"""

import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import src.trading.brokers.kabu.kabu_client  # noqa: F401
from src.trading.brokers.base import BrokerError, OrderSide

_ENV_NAME = "KABU_TRADE_PASSWORD"
_DUMMY = "dummy-trade-pw"


def _make_broker():
    from src.trading.brokers.kabu.kabu_client import KabuBroker

    broker = KabuBroker(api_password="test_password")
    broker._token = "cached_token"
    broker._token_expires_at = datetime.now() + timedelta(hours=1)
    return broker


def _env_without_password() -> dict[str, str]:
    env = dict(os.environ)
    env.pop(_ENV_NAME, None)
    return env


class TestTradePasswordMissing(unittest.TestCase):
    """取引パスワードが未設定・空なら HTTP を送らずに BrokerError"""

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_send_order_raises_when_unset(self, mock_post):
        with patch.dict(os.environ, _env_without_password(), clear=True):
            with self.assertRaises(BrokerError):
                _make_broker().send_order("7203", OrderSide.SELL, 100)
        mock_post.assert_not_called()

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_send_order_raises_when_empty(self, mock_post):
        with patch.dict(os.environ, {_ENV_NAME: ""}):
            with self.assertRaises(BrokerError):
                _make_broker().send_order("7203", OrderSide.BUY, 100)
        mock_post.assert_not_called()

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_send_order_does_not_fetch_token_when_missing(self, mock_post):
        """トークンが無くても、トークン取得の HTTP も送らない"""
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker(api_password="test_password")
        with patch.dict(os.environ, {_ENV_NAME: ""}):
            with self.assertRaises(BrokerError):
                broker.send_order("7203", OrderSide.BUY, 100)
        mock_post.assert_not_called()

    @patch("src.trading.brokers.kabu.kabu_client.httpx.put")
    def test_cancel_order_raises_when_unset(self, mock_put):
        with patch.dict(os.environ, _env_without_password(), clear=True):
            with self.assertRaises(BrokerError):
                _make_broker().cancel_order("ORDER1")
        mock_put.assert_not_called()

    @patch("src.trading.brokers.kabu.kabu_client.httpx.put")
    def test_cancel_order_raises_when_empty(self, mock_put):
        with patch.dict(os.environ, {_ENV_NAME: ""}):
            with self.assertRaises(BrokerError):
                _make_broker().cancel_order("ORDER1")
        mock_put.assert_not_called()


class TestTradePasswordSet(unittest.TestCase):
    """設定されているときは今と同じ body を送る"""

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_send_order_body_unchanged(self, mock_post):
        resp = MagicMock()
        resp.json.return_value = {"Result": 0, "OrderId": "OID-1"}
        mock_post.return_value = resp

        with patch.dict(os.environ, {_ENV_NAME: _DUMMY}):
            _make_broker().send_order("7203.T", OrderSide.BUY, 100)

        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(
            body,
            {
                "Password": _DUMMY,
                "Symbol": "7203",
                "Exchange": 1,
                "SecurityType": 1,
                "Side": "2",
                "CashMargin": 1,
                "DelivType": 2,
                "FundType": "02",
                "AccountType": 4,
                "Qty": 100,
                "FrontOrderType": 10,
                "Price": 0,
                "ExpireDay": 0,
            },
        )

    @patch("src.trading.brokers.kabu.kabu_client.httpx.put")
    def test_cancel_order_body_unchanged(self, mock_put):
        resp = MagicMock()
        resp.json.return_value = {"Result": 0}
        mock_put.return_value = resp

        with patch.dict(os.environ, {_ENV_NAME: _DUMMY}):
            _make_broker().cancel_order("ORDER9")

        body = mock_put.call_args.kwargs["json"]
        self.assertEqual(body, {"OrderId": "ORDER9", "Password": _DUMMY})


if __name__ == "__main__":
    unittest.main()
