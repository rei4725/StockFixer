"""Unit tests for kabu_client.py"""

import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# httpx はオプション依存: テスト環境でインストールされていない場合はモックを設定
if "httpx" not in sys.modules:
    sys.modules["httpx"] = MagicMock()

# Python 3.13+ の patch() は pkgutil.resolve_name を使うため、
# サブパッケージをあらかじめインポートして属性として解決可能にする
import src.trading.brokers.kabu.kabu_client  # noqa: F401

# ──────────────────────────────────────────────────────────────────
# KabuBroker.get_token
# ──────────────────────────────────────────────────────────────────


class TestKabuBrokerGetToken(unittest.TestCase):
    """KabuBroker.get_token のテスト"""

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_returns_token_on_success(self, mock_post):
        """トークン取得成功時にトークン文字列が返ること"""
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        mock_response = MagicMock()
        mock_response.json.return_value = {"Token": "test_token_abc123"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = KabuBroker(api_password="test_password")
        token = broker.get_token()
        self.assertEqual(token, "test_token_abc123")

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_post_called_with_correct_url(self, mock_post):
        """正しいエンドポイントに POST リクエストが送られること"""
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        mock_response = MagicMock()
        mock_response.json.return_value = {"Token": "tok"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = KabuBroker(api_password="test_password")
        broker.get_token()

        call_args = mock_post.call_args
        self.assertIn("/token", call_args[0][0])

    def test_uses_cached_token_when_valid(self):
        """有効なキャッシュトークンがある場合、再取得しないこと"""
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker(api_password="test_password")
        broker._token = "cached_token"
        broker._token_expires_at = datetime.now() + timedelta(hours=1)

        token = broker.get_token()
        self.assertEqual(token, "cached_token")

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_refreshes_token_when_expired(self, mock_post):
        """期限切れのキャッシュトークンは再取得されること"""
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        mock_response = MagicMock()
        mock_response.json.return_value = {"Token": "new_token"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = KabuBroker(api_password="test_password")
        broker._token = "old_token"
        broker._token_expires_at = datetime.now() - timedelta(hours=1)  # 期限切れ

        token = broker.get_token()
        self.assertEqual(token, "new_token")
        mock_post.assert_called_once()

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_refreshes_token_within_buffer_window(self, mock_post):
        """有効期限まで5分未満のトークンは事前更新されること"""
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        mock_response = MagicMock()
        mock_response.json.return_value = {"Token": "refreshed_token"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = KabuBroker(api_password="test_password")
        broker._token = "almost_expired_token"
        broker._token_expires_at = datetime.now() + timedelta(seconds=100)  # バッファ(300s)以内

        token = broker.get_token()
        self.assertEqual(token, "refreshed_token")
        mock_post.assert_called_once()

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_raises_broker_error_after_all_retries_exhausted(self, mock_post):
        """全リトライが失敗したとき BrokerError が raise されること"""
        from src.trading.brokers.base import BrokerError
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        error_response = MagicMock()
        error_response.raise_for_status.side_effect = RuntimeError("HTTP 500")
        mock_post.return_value = error_response

        broker = KabuBroker(api_password="test_password")

        with self.assertRaises(BrokerError):
            broker.get_token()

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_retries_once_on_first_failure_then_succeeds(self, mock_post):
        """1回目の失敗後にリトライして成功すること"""
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        error_response = MagicMock()
        error_response.raise_for_status.side_effect = RuntimeError("timeout")

        success_response = MagicMock()
        success_response.json.return_value = {"Token": "retry_success_token"}
        success_response.raise_for_status = MagicMock()

        mock_post.side_effect = [error_response, success_response]

        broker = KabuBroker(api_password="test_password")
        token = broker.get_token()

        self.assertEqual(token, "retry_success_token")
        self.assertEqual(mock_post.call_count, 2)

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_broker_error_call_count_matches_max_retries(self, mock_post):
        """BrokerError 時の試行回数が _TOKEN_MAX_RETRIES に一致すること"""
        from src.trading.brokers.base import BrokerError
        from src.trading.brokers.kabu.kabu_client import _TOKEN_MAX_RETRIES, KabuBroker

        error_response = MagicMock()
        error_response.raise_for_status.side_effect = RuntimeError("server error")
        mock_post.return_value = error_response

        broker = KabuBroker(api_password="test_password")

        with self.assertRaises(BrokerError):
            broker.get_token()

        self.assertEqual(mock_post.call_count, _TOKEN_MAX_RETRIES)


# ──────────────────────────────────────────────────────────────────
# KabuBroker.get_balance
# ──────────────────────────────────────────────────────────────────


class TestKabuBrokerGetBalance(unittest.TestCase):
    """KabuBroker.get_balance のテスト"""

    def _setup_broker_with_token(self, mock_get, balance_value):
        """キャッシュ済みトークンを持つ broker と残高モックを設定するヘルパー"""
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        mock_response = MagicMock()
        mock_response.json.return_value = {"StockAccountWallet": balance_value}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = KabuBroker(api_password="test_password")
        broker._token = "cached_token"
        broker._token_expires_at = datetime.now() + timedelta(hours=1)
        return broker

    @patch("src.trading.brokers.kabu.kabu_client.httpx.get")
    def test_returns_balance_float(self, mock_get):
        """口座残高が float で返ること"""
        broker = self._setup_broker_with_token(mock_get, 5000000.0)
        balance = broker.get_balance()
        self.assertIsInstance(balance, float)
        self.assertEqual(balance, 5000000.0)

    @patch("src.trading.brokers.kabu.kabu_client.httpx.get")
    def test_returns_zero_when_wallet_missing(self, mock_get):
        """StockAccountWallet キーが無い場合 0.0 が返ること"""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker(api_password="test_password")
        broker._token = "cached_token"
        broker._token_expires_at = datetime.now() + timedelta(hours=1)

        balance = broker.get_balance()
        self.assertEqual(balance, 0.0)

    @patch("src.trading.brokers.kabu.kabu_client.httpx.get")
    def test_get_balance_calls_wallet_endpoint(self, mock_get):
        """正しいエンドポイントに GET リクエストが送られること"""
        broker = self._setup_broker_with_token(mock_get, 1000000.0)
        broker.get_balance()

        call_args = mock_get.call_args
        self.assertIn("wallet", call_args[0][0])


# ──────────────────────────────────────────────────────────────────
# KabuBroker.get_positions
# ──────────────────────────────────────────────────────────────────


class TestKabuBrokerGetPositions(unittest.TestCase):
    """KabuBroker.get_positions のテスト"""

    def _make_broker(self):
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker(api_password="test_password")
        broker._token = "cached_token"
        broker._token_expires_at = datetime.now() + timedelta(hours=1)
        return broker

    @patch("src.trading.brokers.kabu.kabu_client.httpx.get")
    def test_returns_list_of_positions(self, mock_get):
        """ポジション一覧がリストで返ること"""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Symbol": "7203",
                "LeavesQty": 100,
                "Price": 1050.0,
                "CurrentPrice": 1100.0,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker()
        positions = broker.get_positions()

        self.assertIsInstance(positions, list)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "7203")
        self.assertEqual(positions[0]["qty"], 100)

    @patch("src.trading.brokers.kabu.kabu_client.httpx.get")
    def test_returns_empty_list_when_no_positions(self, mock_get):
        """ポジションがない場合は空リストが返ること"""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker()
        positions = broker.get_positions()

        self.assertIsInstance(positions, list)
        self.assertEqual(len(positions), 0)

    @patch("src.trading.brokers.kabu.kabu_client.httpx.get")
    def test_positions_dict_has_required_keys(self, mock_get):
        """ポジション辞書が必須キー（symbol, qty, avg_price, current_price）を持つこと"""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "Symbol": "9984",
                "LeavesQty": 200,
                "Price": 5000.0,
                "CurrentPrice": 5200.0,
            }
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        broker = self._make_broker()
        positions = broker.get_positions()

        pos = positions[0]
        for key in ("symbol", "qty", "avg_price", "current_price"):
            self.assertIn(key, pos)


# ──────────────────────────────────────────────────────────────────
# KabuBroker.cancel_order
# ──────────────────────────────────────────────────────────────────


class TestKabuBrokerCancelOrder(unittest.TestCase):
    """KabuBroker.cancel_order のテスト"""

    def _make_broker(self):
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker(api_password="test_password")
        broker._token = "cached_token"
        broker._token_expires_at = datetime.now() + timedelta(hours=1)
        return broker

    @patch("src.trading.brokers.kabu.kabu_client.httpx.put")
    def test_cancel_order_sends_put_request(self, mock_put):
        """キャンセル注文が PUT リクエストを送ること"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"Result": 0}
        mock_response.raise_for_status = MagicMock()
        mock_put.return_value = mock_response

        broker = self._make_broker()
        result = broker.cancel_order("ORDER123")

        mock_put.assert_called_once()
        self.assertIsInstance(result, dict)

    @patch("src.trading.brokers.kabu.kabu_client.httpx.put")
    def test_cancel_order_includes_order_id_in_result(self, mock_put):
        """返り値に order_id が含まれること"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"Result": 0}
        mock_response.raise_for_status = MagicMock()
        mock_put.return_value = mock_response

        broker = self._make_broker()
        result = broker.cancel_order("ORDER456")

        self.assertEqual(result["order_id"], "ORDER456")
        self.assertEqual(result["status"], "cancelled")

    @patch("src.trading.brokers.kabu.kabu_client.httpx.put")
    def test_cancel_order_url_contains_cancelorder(self, mock_put):
        """PUT リクエストが cancelorder エンドポイントに送られること"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"Result": 0}
        mock_response.raise_for_status = MagicMock()
        mock_put.return_value = mock_response

        broker = self._make_broker()
        broker.cancel_order("ORDER789")

        call_args = mock_put.call_args
        self.assertIn("cancelorder", call_args[0][0])


# ──────────────────────────────────────────────────────────────────
# KabuBroker.send_order
# ──────────────────────────────────────────────────────────────────


class TestKabuBrokerSendOrder(unittest.TestCase):
    """KabuBroker.send_order のテスト"""

    def _make_broker(self):
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker(api_password="test_password")
        broker._token = "cached_token"
        broker._token_expires_at = datetime.now() + timedelta(hours=1)
        return broker

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_send_order_returns_dict_with_order_id(self, mock_post):
        """注文発注が order_id を含む辞書を返すこと"""
        from src.trading.brokers.base import OrderSide

        mock_response = MagicMock()
        mock_response.json.return_value = {"OrderId": "OID-001"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = self._make_broker()
        result = broker.send_order("7203", OrderSide.BUY, 100)

        self.assertIsInstance(result, dict)
        self.assertIn("order_id", result)
        self.assertEqual(result["order_id"], "OID-001")

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_send_order_calls_sendorder_endpoint(self, mock_post):
        """POST リクエストが sendorder エンドポイントに送られること"""
        from src.trading.brokers.base import OrderSide

        mock_response = MagicMock()
        mock_response.json.return_value = {"OrderId": "OID-002"}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        broker = self._make_broker()
        broker.send_order("7203", OrderSide.BUY, 100)

        call_args = mock_post.call_args
        self.assertIn("sendorder", call_args[0][0])


if __name__ == "__main__":
    unittest.main()
