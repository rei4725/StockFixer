"""
ユニットテスト: 閾値の解決やスコアの列の付け足しが例外になっても、損失の止めの決済は走らせる（#932）

_attach_dynamic_thresholds（台帳の読み込み）や apply_multi_horizon_score_column が落ちた日も、
予測を使わない SL/TP の決済と、期限を過ぎた保有のホライズン決済は走らせる。
予測を使う新規の発注は出さず、ジョブは失敗として ERROR を残す。
"""

import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd

from src.trading.brokers.base import BrokerBase, OrderSide
from src.trading.types import TradingGateStatus

_ALLOWED_GATE = TradingGateStatus(
    is_allowed=True,
    stop_active=False,
    reason_code=None,
    reason=None,
    daily_loss=0.0,
    daily_loss_limit=20_000.0,
)

_SL_PCT = 0.05

_PREDICTIONS = pd.DataFrame(
    [
        {
            "market": "jp",
            "symbol": "7000",
            "diff_ratio": 0.05,
            "current_price": 1000.0,
            "predicted_at": "2026-10-24",
        },
        {
            "market": "jp",
            "symbol": "9001",
            "diff_ratio": 0.05,
            "current_price": 1000.0,
            "predicted_at": "2026-10-24",
        },
    ]
)


def _sl_check(avg_price, current_price):
    pnl = (current_price - avg_price) / avg_price
    triggered = pnl <= -_SL_PCT
    return triggered, False, "SL発動" if triggered else ""


def _make_broker():
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = [
        {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 900.0},
        {"symbol": "9001", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
    ]
    broker.get_short_positions = MagicMock(return_value=[])
    broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
    return broker


class TestThresholdErrorStillExits(unittest.TestCase):
    def setUp(self):
        from src.orchestration.jobs import daily

        self.daily = daily
        self.broker = _make_broker()
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.stack = stack

        stack.enter_context(
            patch(
                "src.trading.brokers.account.factory.make_order_broker",
                return_value=self.broker,
            )
        )
        repo_cls = stack.enter_context(
            patch(
                "src.infrastructure.persistence.prediction_result_repository."
                "PostgresPredictionResultRepository"
            )
        )
        repo_cls.return_value.get_latest_by_market.return_value = _PREDICTIONS.copy()
        for target in (
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter",
            "src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter",
            "src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink",
            "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink",
            "src.trading.execution.runner._record_order",
            "src.trading.execution.sl_tp._record_order",
            "src.trading.execution.RiskManager.update_peak_balance",
            "src.reporting.discord.discord_utils.send_daily_order_completion",
            "src.reporting.discord.discord_utils.send_correlation_alert",
        ):
            stack.enter_context(patch(target))
        sl_risk = stack.enter_context(patch("src.trading.execution.sl_tp.RiskManager"))
        sl_risk.return_value.check_sl_tp.side_effect = _sl_check
        stack.enter_context(
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=_ALLOWED_GATE,
            )
        )
        stack.enter_context(
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=100)
        )
        stack.enter_context(patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True))
        stack.enter_context(patch.object(daily, "record_loss_limit_streak"))
        self.horizon_exit = stack.enter_context(patch.object(daily, "run_horizon_exit_check"))

    def _run_and_check(self):
        with self.assertLogs("src.trading.execution.runner", level="ERROR"):
            with self.assertLogs("src.orchestration.jobs.daily", level="ERROR"):
                with self.assertRaises(RuntimeError):
                    self.daily.run_daily_auto_order()

        sent = [(c.args[0], c.args[1]) for c in self.broker.send_order.call_args_list]
        # SL の線を割った保有だけを売り、新しい買い・ショートは出さない
        self.assertEqual(sent, [("9000", OrderSide.SELL)])
        self.horizon_exit.assert_called_once_with()

    def test_attach_dynamic_thresholds_error(self):
        self.stack.enter_context(
            patch(
                "src.trading.execution.runner._attach_dynamic_thresholds",
                side_effect=RuntimeError("optimal_params.json が読めない"),
            )
        )
        self._run_and_check()

    def test_multi_horizon_score_error(self):
        self.stack.enter_context(
            patch(
                "src.trading.execution.runner.apply_multi_horizon_score_column",
                side_effect=RuntimeError("score failed"),
            )
        )
        self._run_and_check()


if __name__ == "__main__":
    unittest.main()
