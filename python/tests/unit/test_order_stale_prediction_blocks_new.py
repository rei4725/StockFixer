"""8:50 の自動発注は、今日（JST）の production の予想の保存が終わっているときだけ新規を出す（#1394）。

日次の道が遅れた朝やり直した朝は、最新の回が前の日の回か、今日の回の保存の途中になる。
そのときは新規の買い・ショートを止め、SL/TP の決済は今までどおり出し、理由を1回知らせる。
"""

import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from src.infrastructure.in_memory import (
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import run_daily_orders
from src.trading.types import CorrelationGateResult, TradingGateStatus

_PREDICTIONS = InMemoryPredictionRepository()

# 8:50 JST の発注の時刻（コンテナの時計は UTC）
_NOW_UTC = datetime(2026, 10, 29, 23, 50, tzinfo=timezone.utc)
_TODAY_JST = "20261030"
_TODAY_RUN = "20261029_224343"  # 7:43 JST の今日の回
_YESTERDAY_RUN = "20261028_224343"  # 前の日の回

_OPEN_GATE = TradingGateStatus(
    is_allowed=True, stop_active=False, daily_loss=0.0, daily_loss_limit=20_000.0
)


def _row(symbol, diff_ratio, predicted_at):
    return {
        "market": "jp",
        "symbol": symbol,
        "predicted_at": predicted_at,
        "current_price": 1000.0,
        "diff_ratio": diff_ratio,
        "confidence_ratio": 1.0,
    }


def _sl_check(avg_price, current_price):
    triggered = (current_price - avg_price) / avg_price <= -0.05
    return triggered, False, "SL発動" if triggered else ""


class TestStalePredictionBlocksNew(unittest.TestCase):
    def _run(self, predicted_at, saved_day, short_side=False):
        positions = [{"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 900.0}]
        predictions = pd.DataFrame(
            [_row("7200", 0.03, predicted_at), _row("8001", -0.03, predicted_at)]
        )
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        broker.get_balance.return_value = 1_000_000.0
        broker.get_positions.return_value = positions
        broker.get_short_positions = MagicMock(return_value=[])
        broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
        sl_risk = MagicMock()
        sl_risk.return_value.check_sl_tp.side_effect = _sl_check
        with ExitStack() as stack:

            def p(*args, **kwargs):
                return stack.enter_context(patch(*args, **kwargs))

            stack.enter_context(
                patch.object(_PREDICTIONS, "get_latest_by_market", return_value=predictions)
            )
            p("src.trading.execution.freshness._local_now", return_value=_NOW_UTC)
            p("src.trading.execution.runner.ENABLE_SHORT_SIDE", short_side)
            p("src.trading.execution.runner._record_order")
            p("src.trading.execution.sl_tp._record_order")
            p(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            )
            p("src.trading.execution.sl_tp.RiskManager", sl_risk)
            p("src.trading.execution.RiskManager.evaluate_trading_gate", return_value=_OPEN_GATE)
            p("src.trading.execution.RiskManager.calc_position_size", return_value=100)
            p("src.trading.execution.RiskManager.update_peak_balance")
            p(
                "src.trading.execution.runner.evaluate_correlation_gate",
                return_value=CorrelationGateResult(
                    is_allowed=True, enc=2.0, enc_threshold=2.0, avg_correlation=0.0, n_symbols=1
                ),
            )
            p("src.trading.execution.runner._load_exit_model", return_value=None)
            p("src.trading.execution.params.get_optimal_params", return_value={})
            p(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            )
            stats = run_daily_orders(
                broker,
                order_run_sink=InMemoryOrderRunSink(),
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
                market="jp",
                mode="paper",
                prediction_saved_day=saved_day,
            )
        sent = [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]
        return stats, sent

    def test_previous_day_run_blocks_new_and_keeps_sl(self):
        """最新の回が前の日（JST）なら新規の買いを出さず、SL の決済は出す"""
        stats, sent = self._run(_YESTERDAY_RUN, _TODAY_JST)
        self.assertEqual(sent, [("9000", OrderSide.SELL)])
        self.assertEqual(stats["buy_orders"], 0)
        self.assertIn(_YESTERDAY_RUN, stats["stale_prediction_reason"])

    def test_previous_day_run_blocks_new_short(self):
        """最新の回が前の日なら新規のショートも出さない"""
        stats, sent = self._run(_YESTERDAY_RUN, _TODAY_JST, short_side=True)
        self.assertEqual(sent, [("9000", OrderSide.SELL)])
        self.assertEqual(stats["short_orders"], 0)

    def test_today_rows_without_saved_marker_block_new(self):
        """今日の回の行はあるが保存の終わりの印が無い（保存の途中）なら新規を出さない"""
        for saved_day in ("", "20261029"):
            with self.subTest(saved_day=saved_day):
                stats, sent = self._run(_TODAY_RUN, saved_day, short_side=True)
                self.assertEqual(sent, [("9000", OrderSide.SELL)])
                self.assertEqual(stats["buy_orders"], 0)
                self.assertEqual(stats["short_orders"], 0)
                self.assertIsNotNone(stats["stale_prediction_reason"])

    def test_today_run_saved_keeps_new_orders(self):
        """今日の回が保存し終わっている朝は、今までどおり新規を出す"""
        stats, sent = self._run(_TODAY_RUN, _TODAY_JST, short_side=True)
        self.assertIn(("9000", OrderSide.SELL), sent)
        self.assertIn(("7200", OrderSide.BUY), sent)
        self.assertIn(("8001", OrderSide.SHORT), sent)
        self.assertEqual(stats["buy_orders"], 1)
        self.assertIsNone(stats["stale_prediction_reason"])


class TestPredictionSavedMarker(unittest.TestCase):
    def test_mark_then_clear(self):
        """[3/6] の保存の後に今日の JST の日付を残し、[3/6] の始まりで消す"""
        from src.orchestration.jobs import prediction_saved_marker as marker

        with patch("src.orchestration.jobs.common._local_now", return_value=_NOW_UTC):
            marker.mark_prediction_saved()
            self.assertEqual(marker.read_prediction_saved_day(), _TODAY_JST)
            marker.clear_prediction_saved_marker()
            self.assertEqual(marker.read_prediction_saved_day(), "")


class TestAutoOrderNotifiesStale(unittest.TestCase):
    def _run(self, stats):
        from src.orchestration.jobs import daily

        with ExitStack() as stack:

            def p(*args, **kwargs):
                return stack.enter_context(patch(*args, **kwargs))

            run_orders = p("src.trading.execution.run_daily_orders", return_value=stats)
            p("src.trading.brokers.account.factory.make_order_broker")
            notifier = p(
                "src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter"
            )
            for target in (
                "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter",
                "src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink",
                "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink",
                "src.infrastructure.persistence.prediction_result_repository."
                "PostgresPredictionResultRepository",
                "src.reporting.discord.discord_utils.send_daily_order_completion",
            ):
                p(target)
            p(
                "src.orchestration.jobs.daily.read_prediction_saved_day",
                return_value=_TODAY_JST,
            )
            stack.enter_context(patch.object(daily, "record_loss_limit_streak"))
            stack.enter_context(patch.object(daily, "run_horizon_exit_check"))
            daily.run_daily_auto_order()
        return run_orders, notifier.return_value.send_alert

    def test_stale_reason_notified_once(self):
        """新規を止めた日は、件数と理由を1回知らせ、印の日付を発注へ渡す"""
        stats = {
            "buy_orders": 0,
            "sell_orders": 1,
            "errors": 0,
            "stale_prediction_reason": "最新の回が前の日",
        }
        run_orders, send_alert = self._run(stats)
        self.assertEqual(run_orders.call_args.kwargs["prediction_saved_day"], _TODAY_JST)
        send_alert.assert_called_once()
        self.assertIn("最新の回が前の日", send_alert.call_args.args[1])

    def test_fresh_day_not_notified(self):
        """今日の回が保存し終わっている日は、止めた知らせを出さない"""
        stats = {"buy_orders": 1, "sell_orders": 0, "errors": 0, "stale_prediction_reason": None}
        _, send_alert = self._run(stats)
        send_alert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
