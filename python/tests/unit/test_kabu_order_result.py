"""KabuBroker の発注・取消の応答の Result と OrderId の確かめのテスト（Issue 951）"""

import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import src.trading.brokers.kabu.kabu_client  # noqa: F401
from src.trading.brokers.base import BrokerError, OrderSide

_ENV = {"KABU_TRADE_PASSWORD": "dummy-trade-pw"}


def _make_broker():
    from src.trading.brokers.kabu.kabu_client import KabuBroker

    broker = KabuBroker(api_password="test_password")
    broker._token = "cached_token"
    broker._token_expires_at = datetime.now() + timedelta(hours=1)
    return broker


def _resp(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


class TestSendOrderResult(unittest.TestCase):
    """send_order は Result が 0 でない・OrderId が空の応答を断られた注文とする"""

    def _send(self, payload):
        with patch("src.trading.brokers.kabu.kabu_client.httpx.post") as mock_post:
            mock_post.return_value = _resp(payload)
            with patch.dict(os.environ, _ENV):
                return _make_broker().send_order("7203", OrderSide.BUY, 100)

    def test_nonzero_result_raises(self):
        with self.assertRaises(BrokerError) as ctx:
            self._send({"Result": 4, "Message": "rejected"})
        self.assertIn("rejected", str(ctx.exception))

    def test_empty_response_raises(self):
        with self.assertRaises(BrokerError):
            self._send({})

    def test_empty_order_id_raises(self):
        with self.assertRaises(BrokerError):
            self._send({"Result": 0, "OrderId": ""})

    def test_missing_result_raises(self):
        with self.assertRaises(BrokerError):
            self._send({"OrderId": "X"})

    def test_accepted_returns_order_id(self):
        result = self._send({"Result": 0, "OrderId": "X"})
        self.assertEqual(result["order_id"], "X")
        self.assertEqual(result["status"], "accepted")


class TestCancelOrderResult(unittest.TestCase):
    """cancel_order は Result が 0 でない応答を断られた取消とする"""

    def _cancel(self, payload):
        with patch("src.trading.brokers.kabu.kabu_client.httpx.put") as mock_put:
            mock_put.return_value = _resp(payload)
            with patch.dict(os.environ, _ENV):
                return _make_broker().cancel_order("ORDER1")

    def test_nonzero_result_raises(self):
        with self.assertRaises(BrokerError) as ctx:
            self._cancel({"Result": 43, "Message": "already done"})
        self.assertIn("already done", str(ctx.exception))

    def test_empty_response_raises(self):
        with self.assertRaises(BrokerError):
            self._cancel({})

    def test_zero_result_returns_cancelled(self):
        result = self._cancel({"Result": 0, "OrderId": "ORDER1"})
        self.assertEqual(result["order_id"], "ORDER1")
        self.assertEqual(result["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
