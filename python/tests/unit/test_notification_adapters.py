"""ユニットテスト: 通知アダプター（NotificationPort 実装群）"""

import logging

from src.domain.ports import AlertLevel, NotificationPort
from src.infrastructure.in_memory import InMemoryNotificationAdapter, NullNotificationAdapter
from src.infrastructure.log_notification_adapter import LogNotificationAdapter

# ---------------------------------------------------------------------------
# NullNotificationAdapter
# ---------------------------------------------------------------------------


class TestNullNotificationAdapter:
    def test_implements_notification_port(self):
        adapter = NullNotificationAdapter()
        assert isinstance(adapter, NotificationPort)

    def test_send_message_returns_true(self):
        adapter = NullNotificationAdapter()
        assert adapter.send_message("hello") is True

    def test_send_alert_returns_true(self):
        adapter = NullNotificationAdapter()
        assert adapter.send_alert("title", "message", AlertLevel.ERROR) is True

    def test_send_alert_with_default_level(self):
        adapter = NullNotificationAdapter()
        assert adapter.send_alert("title", "message") is True


# ---------------------------------------------------------------------------
# LogNotificationAdapter
# ---------------------------------------------------------------------------


class TestLogNotificationAdapter:
    def test_implements_notification_port(self):
        adapter = LogNotificationAdapter()
        assert isinstance(adapter, NotificationPort)

    def test_send_message_returns_true(self, caplog):
        adapter = LogNotificationAdapter()
        with caplog.at_level(logging.INFO):
            result = adapter.send_message("test message")
        assert result is True
        assert "test message" in caplog.text

    def test_send_alert_info_returns_true(self, caplog):
        adapter = LogNotificationAdapter()
        with caplog.at_level(logging.INFO):
            result = adapter.send_alert("Info Title", "info body", AlertLevel.INFO)
        assert result is True
        assert "Info Title" in caplog.text

    def test_send_alert_warning_uses_warning_level(self, caplog):
        adapter = LogNotificationAdapter()
        with caplog.at_level(logging.WARNING):
            adapter.send_alert("Warn Title", "warn body", AlertLevel.WARNING)
        assert "Warn Title" in caplog.text

    def test_send_alert_error_uses_error_level(self, caplog):
        adapter = LogNotificationAdapter()
        with caplog.at_level(logging.ERROR):
            adapter.send_alert("Error Title", "error body", AlertLevel.ERROR)
        assert "Error Title" in caplog.text

    def test_send_alert_default_level_is_info(self, caplog):
        adapter = LogNotificationAdapter()
        with caplog.at_level(logging.INFO):
            result = adapter.send_alert("Title", "body")
        assert result is True


# ---------------------------------------------------------------------------
# InMemoryNotificationAdapter — send_alert の追加分
# ---------------------------------------------------------------------------


class TestInMemoryNotificationAdapterSendAlert:
    def test_send_alert_stores_alert(self):
        adapter = InMemoryNotificationAdapter()
        adapter.send_alert("Token Error", "Failed to get token", AlertLevel.ERROR)
        assert len(adapter.sent_alerts) == 1
        assert adapter.sent_alerts[0]["title"] == "Token Error"
        assert adapter.sent_alerts[0]["level"] == AlertLevel.ERROR

    def test_send_alert_fails_when_should_fail(self):
        adapter = InMemoryNotificationAdapter()
        adapter.should_fail = True
        result = adapter.send_alert("Title", "msg", AlertLevel.WARNING)
        assert result is False
        assert len(adapter.sent_alerts) == 0

    def test_send_alert_does_not_affect_sent_messages(self):
        adapter = InMemoryNotificationAdapter()
        adapter.send_alert("T", "M", AlertLevel.INFO)
        assert len(adapter.sent_messages) == 0


# ---------------------------------------------------------------------------
# DI: run_daily_orders が notifier.send_alert を呼ぶことを確認
# ---------------------------------------------------------------------------


class TestRunDailyOrdersNotifierDI:
    def test_notifier_send_alert_called_on_broker_error(self):
        from unittest.mock import MagicMock

        from src.trading.brokers.base import BrokerBase, BrokerError
        from src.trading.execution import run_daily_orders

        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        broker.get_token.side_effect = BrokerError("token error")

        notifier = InMemoryNotificationAdapter()

        stats = run_daily_orders(broker, market="jp", mode="paper", notifier=notifier)

        assert len(notifier.sent_alerts) == 1
        assert notifier.sent_alerts[0]["level"] == AlertLevel.ERROR
        assert stats["buy_orders"] == 0

    def test_no_alert_when_notifier_is_none_and_broker_error(self):
        from unittest.mock import MagicMock

        from src.trading.brokers.base import BrokerBase, BrokerError
        from src.trading.execution import run_daily_orders

        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        broker.get_token.side_effect = BrokerError("token error")

        stats = run_daily_orders(broker, market="jp", mode="paper", notifier=None)

        assert stats["buy_orders"] == 0

    def test_null_adapter_suppresses_alert(self):
        from unittest.mock import MagicMock

        from src.trading.brokers.base import BrokerBase, BrokerError
        from src.trading.execution import run_daily_orders

        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        broker.get_token.side_effect = BrokerError("token error")

        notifier = NullNotificationAdapter()
        stats = run_daily_orders(broker, market="jp", mode="paper", notifier=notifier)

        assert stats["buy_orders"] == 0
