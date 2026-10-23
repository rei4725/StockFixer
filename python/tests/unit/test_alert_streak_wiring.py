"""運用アラートの連続回数（NF-303-2・3）が日次の道の結果で更新されることのテスト（#934）。"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd

from src.utils.alert_service import (
    DRIFT_WARN_THRESHOLD,
    LOSS_LIMIT_THRESHOLD,
    check_drift_warn_rule,
    check_loss_limit_rule,
)

_KEY_LOSS = "alert.loss_limit_streak"
_KEY_DRIFT = "alert.drift_warn_streak"


class _ConfigStoreTestCase(unittest.TestCase):
    """system_config を辞書に置き換える。"""

    def setUp(self):
        self.store: dict[str, str] = {}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            patch(
                "src.utils.db.system_config.get_config_value",
                side_effect=lambda key, default=None: self.store.get(key, default),
            )
        )
        stack.enter_context(
            patch(
                "src.utils.db.system_config.set_config_value",
                side_effect=lambda key, value: self.store.__setitem__(key, value),
            )
        )


class TestLossLimitStreakWiring(_ConfigStoreTestCase):
    def _run_auto_order(self, reason_code):
        from src.orchestration.jobs import daily

        stats = {
            "buy_orders": 0,
            "sell_orders": 0,
            "trading_stopped": reason_code is not None,
            "reason_code": reason_code,
        }
        with patch("src.trading.execution.run_daily_orders", return_value=stats), patch(
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"
        ), patch("src.trading.brokers.paper.paper_broker.PaperBroker"), patch(
            "src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink"
        ), patch(
            "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"
        ), patch(
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository"
        ), patch(
            "src.reporting.discord.discord_utils.send_daily_order_completion"
        ), patch.object(
            daily, "run_horizon_exit_check"
        ):
            daily.run_daily_auto_order()

    def test_loss_limit_streak_counts_up_and_triggers(self):
        for _ in range(LOSS_LIMIT_THRESHOLD - 1):
            self._run_auto_order("daily_loss_limit")
        self.assertEqual(int(self.store.get(_KEY_LOSS, "0")), LOSS_LIMIT_THRESHOLD - 1)
        self.assertFalse(check_loss_limit_rule().triggered)

        self._run_auto_order("daily_loss_limit")
        result = check_loss_limit_rule()
        self.assertTrue(result.triggered)
        self.assertEqual(result.rule_id, "NF-303-2")
        self.assertEqual(result.consecutive_count, LOSS_LIMIT_THRESHOLD)

    def test_loss_limit_streak_resets_on_normal_day(self):
        self.store[_KEY_LOSS] = "2"
        self._run_auto_order(None)
        self.assertEqual(self.store[_KEY_LOSS], "0")

    def test_other_stop_reason_does_not_count(self):
        self.store[_KEY_LOSS] = "2"
        self._run_auto_order("consecutive_losses")
        self.assertEqual(self.store[_KEY_LOSS], "0")

    def test_streak_update_failure_does_not_stop_order_job(self):
        with patch(
            "src.orchestration.jobs.alert_streaks.update_loss_limit_streak",
            side_effect=RuntimeError("db down"),
        ):
            with self.assertLogs("src.orchestration.jobs.alert_streaks", level="ERROR"):
                self._run_auto_order("daily_loss_limit")


class TestDriftWarnStreakWiring(_ConfigStoreTestCase):
    def _run_drift_check(self, summary):
        from src.orchestration.jobs.drift_check import run_daily_drift_check

        with patch("src.prediction.db.load_drift_summary", return_value=summary), patch(
            "src.reporting.discord.discord_utils.send_drift_retrain_notification"
        ), patch("src.watchlist.batch_runner.load_target_symbols", return_value=[]), patch(
            "src.prediction.training_pipeline.train_models_for_symbol_task"
        ):
            run_daily_drift_check()

    @staticmethod
    def _summary(mae: float, hit_rate: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "mean_abs_error": mae,
                    "direction_accuracy": hit_rate,
                    "n_samples": 20,
                }
            ]
        )

    def test_drift_warn_streak_counts_up_and_triggers(self):
        drifted = self._summary(mae=0.05, hit_rate=0.3)
        for _ in range(DRIFT_WARN_THRESHOLD - 1):
            self._run_drift_check(drifted)
        self.assertEqual(int(self.store.get(_KEY_DRIFT, "0")), DRIFT_WARN_THRESHOLD - 1)
        self.assertFalse(check_drift_warn_rule().triggered)

        self._run_drift_check(drifted)
        result = check_drift_warn_rule()
        self.assertTrue(result.triggered)
        self.assertEqual(result.rule_id, "NF-303-3")
        self.assertEqual(result.consecutive_count, DRIFT_WARN_THRESHOLD)

    def test_drift_warn_streak_resets_when_no_drift(self):
        self.store[_KEY_DRIFT] = "1"
        self._run_drift_check(self._summary(mae=0.001, hit_rate=0.9))
        self.assertEqual(self.store[_KEY_DRIFT], "0")

    def test_drift_warn_streak_unchanged_when_no_accuracy_data(self):
        self.store[_KEY_DRIFT] = "1"
        self._run_drift_check(pd.DataFrame())
        self.assertEqual(self.store[_KEY_DRIFT], "1")


if __name__ == "__main__":
    unittest.main()
