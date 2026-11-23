"""自動発注の後の知らせ（auto_order_notify）の直のテスト（#2096）。"""

from unittest.mock import patch

from config.settings import CORRELATION_ENC_THRESHOLD
from src.orchestration.jobs.auto_order_notify import (
    notify_auto_order_completion,
    notify_correlation_blocked,
)

_COMPLETION = "src.reporting.discord.discord_utils.send_daily_order_completion"
_CORRELATION = "src.reporting.discord.discord_utils.send_correlation_alert"
_TRADE_LINES = "src.orchestration.jobs.trade_lines.order_trade_lines"


class TestNotifyAutoOrderCompletion:
    def test_passes_stats_and_defaults(self):
        stats = {"buy_orders": 2, "sell_orders": 1}
        with (
            patch(_COMPLETION) as send,
            patch(_TRADE_LINES, return_value=["line"]) as lines,
        ):
            notify_auto_order_completion(stats, "paper")
        lines.assert_called_once_with(stats)
        send.assert_called_once_with(
            buy_orders=2,
            sell_orders=1,
            mode="paper",
            trading_stopped=False,
            stop_reason=None,
            daily_loss=None,
            daily_loss_limit=None,
            errors=0,
            sl_tp_unevaluable_symbols=None,
            skipped=0,
            skip_reasons=None,
            details=["line"],
        )

    def test_passes_given_values(self):
        stats = {
            "buy_orders": 0,
            "sell_orders": 3,
            "trading_stopped": True,
            "stop_reason": "loss",
            "daily_loss": -100.0,
            "daily_loss_limit": 200.0,
            "errors": 4,
            "sl_tp_unevaluable_symbols": ["7203"],
            "skipped": 230,
            "skip_reasons": {"sector_cap": 208, "max_orders_per_run": 22},
        }
        with patch(_COMPLETION) as send, patch(_TRADE_LINES, return_value=[]):
            notify_auto_order_completion(stats, "paper")
        kwargs = send.call_args.kwargs
        assert kwargs["skipped"] == 230
        assert kwargs["skip_reasons"] == {"sector_cap": 208, "max_orders_per_run": 22}
        assert kwargs["trading_stopped"] is True
        assert kwargs["stop_reason"] == "loss"
        assert kwargs["daily_loss"] == -100.0
        assert kwargs["daily_loss_limit"] == 200.0
        assert kwargs["errors"] == 4
        assert kwargs["sl_tp_unevaluable_symbols"] == ["7203"]

    def test_send_failure_does_not_raise(self):
        stats = {"buy_orders": 1, "sell_orders": 0}
        with (
            patch(_COMPLETION, side_effect=RuntimeError("boom")),
            patch(_TRADE_LINES, return_value=[]),
        ):
            notify_auto_order_completion(stats, "paper")

    def test_missing_key_does_not_raise(self):
        with patch(_COMPLETION) as send:
            notify_auto_order_completion({}, "paper")
        send.assert_not_called()


class TestNotifyCorrelationBlocked:
    def test_not_blocked_sends_nothing(self):
        with patch(_CORRELATION) as send:
            notify_correlation_blocked({"correlation_blocked": False})
            notify_correlation_blocked({})
        send.assert_not_called()

    def test_blocked_uses_defaults(self):
        with patch(_CORRELATION) as send:
            notify_correlation_blocked({"correlation_blocked": True})
        send.assert_called_once_with(
            enc=0.0,
            enc_threshold=CORRELATION_ENC_THRESHOLD,
            avg_correlation=0.0,
            n_symbols=0,
            symbols=[],
        )

    def test_blocked_passes_values(self):
        stats = {
            "correlation_blocked": True,
            "enc": 1.5,
            "avg_correlation": 0.8,
            "n_held_symbols": 2,
            "held_symbols_list": ["7203", "6758"],
        }
        with patch(_CORRELATION) as send:
            notify_correlation_blocked(stats)
        send.assert_called_once_with(
            enc=1.5,
            enc_threshold=CORRELATION_ENC_THRESHOLD,
            avg_correlation=0.8,
            n_symbols=2,
            symbols=["7203", "6758"],
        )

    def test_send_failure_does_not_raise(self):
        with patch(_CORRELATION, side_effect=RuntimeError("boom")):
            notify_correlation_blocked({"correlation_blocked": True})
