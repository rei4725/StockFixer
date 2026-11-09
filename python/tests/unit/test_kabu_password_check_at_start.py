"""発注の回の始めで KabuBroker のパスワードの欠けを1回だけ確かめるテスト（Issue 1757）。

取引パスワード・API パスワードが欠けていれば、注文ごとに BrokerError を並べず、
回の始めで知らせを1通出してその回の発注を止める。API パスワードの欠けも BrokerError。
"""

import os
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import src.trading.brokers.kabu.kabu_client  # noqa: F401
from src.domain.ports import AlertLevel
from src.infrastructure.in_memory import (
    InMemoryNotificationAdapter,
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerError

_TRADE_ENV = "KABU_TRADE_PASSWORD"
_API_ENV = "KABU_API_PASSWORD"


def _kabu_broker(api_password: str = "test_password"):
    from src.trading.brokers.kabu.kabu_client import KabuBroker

    broker = KabuBroker(api_password=api_password)
    broker._token = "cached_token"
    broker._token_expires_at = datetime.now() + timedelta(hours=1)
    return broker


def _env_without(*names: str) -> dict[str, str]:
    env = dict(os.environ)
    for name in names:
        env.pop(name, None)
    return env


class TestOrderRunStopsWhenTradePasswordMissing(unittest.TestCase):
    """取引パスワードが空なら、回の始めで1通知らせて send_order を1回も呼ばない"""

    def _run(self, broker, notifier):
        from src.trading.execution import run_daily_orders

        return run_daily_orders(
            broker,
            order_run_sink=InMemoryOrderRunSink(),
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=InMemoryPredictionRepository(),
            market="jp",
            mode="paper",
            notifier=notifier,
        )

    @patch("src.trading.brokers.kabu.kabu_client.httpx.get")
    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_trade_password_empty_stops_run_with_one_alert(self, mock_post, mock_get):
        broker = _kabu_broker()
        notifier = InMemoryNotificationAdapter()

        with (
            patch.dict(os.environ, {_TRADE_ENV: ""}),
            patch.object(broker, "send_order") as send_order,
            patch("src.trading.execution.gate_record._write_record"),
            self.assertLogs(level="ERROR"),
        ):
            stats = self._run(broker, notifier)

        send_order.assert_not_called()
        mock_post.assert_not_called()
        mock_get.assert_not_called()
        self.assertEqual([a["level"] for a in notifier.sent_alerts], [AlertLevel.ERROR])
        self.assertIn(_TRADE_ENV, notifier.sent_alerts[0]["message"])
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["buy_orders"], 0)
        self.assertEqual(stats["sell_orders"], 0)


class TestCheckOrderReady(unittest.TestCase):
    """KabuBroker.check_order_ready は HTTP を送らずに欠けを BrokerError で知らせる"""

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_trade_password_empty_raises(self, mock_post):
        with patch.dict(os.environ, {_TRADE_ENV: ""}):
            with self.assertRaises(BrokerError):
                _kabu_broker().check_order_ready()
        mock_post.assert_not_called()

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_api_password_empty_raises(self, mock_post):
        with patch.dict(os.environ, _env_without(_API_ENV), clear=True):
            os.environ[_TRADE_ENV] = "dummy-trade-pw"
            with self.assertRaises(BrokerError):
                _kabu_broker(api_password="").check_order_ready()
        mock_post.assert_not_called()

    def test_both_set_passes(self):
        with patch.dict(os.environ, {_TRADE_ENV: "dummy-trade-pw"}):
            _kabu_broker().check_order_ready()

    def test_base_default_is_noop(self):
        from src.trading.brokers.paper.paper_broker import PaperBroker

        self.assertIsNone(PaperBroker.check_order_ready(MagicMock(spec=PaperBroker)))


class TestApiPasswordMissingIsBrokerError(unittest.TestCase):
    """API パスワードの欠けも、取引パスワードと同じく BrokerError で止める"""

    @patch("src.trading.brokers.kabu.kabu_client.httpx.post")
    def test_get_token_raises_broker_error(self, mock_post):
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        with patch.dict(os.environ, _env_without(_API_ENV), clear=True):
            broker = KabuBroker(api_password="")
            with self.assertRaises(BrokerError):
                broker.get_token()
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
