"""NF-303-3 のドリフト Warn を閾値超えの割合で数え、同じ状態で毎日鳴らさないことのテスト（#1369）。"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd

from src.utils.alert_service import (
    DRIFT_RENOTIFY_DAYS,
    DRIFT_WARN_THRESHOLD,
    check_drift_warn_rule,
    decide_drift_notice,
)

_KEY_DRIFT = "alert.drift_warn_streak"


def _row(symbol: str, mae: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "mean_abs_error": mae,
        "direction_accuracy": 0.55,
        "n_samples": 20,
    }


def _summary(n_bad: int, n_total: int = 20) -> pd.DataFrame:
    """n_total 銘柄のうち n_bad 銘柄だけ MAE が中央値×1.5 を超える表。"""
    rows = [_row(f"BAD{i}", 0.20) for i in range(n_bad)]
    rows += [_row(f"OK{i}", 0.03) for i in range(n_total - n_bad)]
    return pd.DataFrame(rows)


class _StoreTestCase(unittest.TestCase):
    """system_config を辞書に置き換え、記録のたびに JST の日を1日進める。"""

    def setUp(self):
        self.store: dict[str, str] = {}
        stack = ExitStack()
        self.addCleanup(stack.close)
        days = (f"2026{m:02d}{d:02d}" for m in (10, 11) for d in range(1, 31))
        stack.enter_context(
            patch(
                "src.orchestration.jobs.alert_streaks.today_jst_key",
                side_effect=lambda: next(days),
            )
        )
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
        self.train = stack.enter_context(
            patch("src.prediction.training_pipeline.train_models_for_symbol_task")
        )
        self.notify = stack.enter_context(
            patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
        )
        stack.enter_context(
            patch("src.orchestration.jobs.drift_check._is_recently_retrained", return_value=False)
        )

    def _run_day(self, n_bad: int, n_total: int = 20) -> None:
        from src.domain.types import SymbolTask
        from src.orchestration.jobs.drift_check import run_daily_drift_check

        summary = _summary(n_bad, n_total)
        tasks = [SymbolTask(market="jp", symbol=s) for s in summary["symbol"]]
        self.train.return_value = {"status": "success"}
        with patch("src.prediction.db.load_drift_summary", return_value=summary), patch(
            "src.watchlist.batch_runner.load_target_symbols", return_value=tasks
        ), patch("src.prediction.db.load_feature_exclusion_candidates", return_value=[]), patch(
            "src.reporting.discord.discord_utils.send_feature_suggestion_notification"
        ):
            run_daily_drift_check()

    def _streak(self) -> int:
        return int(self.store.get(_KEY_DRIFT, "0"))


class TestWarnByShare(_StoreTestCase):
    def test_share_below_line_resets_streak(self):
        """閾値超えが1銘柄以上でも、割合が線（既定 15%）より小さい日は 0 に戻る。"""
        self.store[_KEY_DRIFT] = "3"
        self._run_day(n_bad=2)  # 10%
        self.assertEqual(self._streak(), 0)

    def test_share_at_or_above_line_counts_up(self):
        self._run_day(n_bad=3)  # 15%
        self.assertEqual(self._streak(), 1)
        self._run_day(n_bad=10)  # 50%
        self.assertEqual(self._streak(), 2)
        self._run_day(n_bad=1)  # 5%
        self.assertEqual(self._streak(), 0)

    def test_retrain_still_runs_below_line(self):
        """割合が線より小さく Warn にならない日も、閾値超えの銘柄は今までどおり再学習に回す。"""
        self._run_day(n_bad=1)
        self.assertEqual(self._streak(), 0)
        self.train.assert_called_once()
        self.assertEqual(self.train.call_args.args[0].symbol, "BAD0")
        self.notify.assert_called_once()

    def test_line_comes_from_system_config(self):
        self.store["drift.warn_share"] = "0.05"
        self._run_day(n_bad=1)  # 5%
        self.assertEqual(self._streak(), 1)

    def test_share_count_and_denominator_are_recorded(self):
        self._run_day(n_bad=4)
        self._run_day(n_bad=4)
        result = check_drift_warn_rule()
        self.assertTrue(result.triggered)
        text = "\n".join(result.as_discord_lines())
        self.assertIn("閾値超過 4 / 判定 20 銘柄（割合 20.0%）", text)


class TestNoticeOnlyOnChange(_StoreTestCase):
    def _day(self, n_bad: int) -> str:
        self._run_day(n_bad)
        result = check_drift_warn_rule()
        return result.details["notice"] if result.triggered else ""

    def test_same_share_is_not_repeated_until_renotify_days(self):
        notices = [self._day(4) for _ in range(DRIFT_WARN_THRESHOLD + DRIFT_RENOTIFY_DAYS + 1)]
        self.assertEqual(notices[DRIFT_WARN_THRESHOLD - 1], "start")
        # 始まった日の後は、決めた日数が経つまで同じ割合では知らせない
        start = DRIFT_WARN_THRESHOLD - 1
        self.assertEqual(notices[:start], [""] * start)
        self.assertEqual(notices[start + 1 : start + DRIFT_RENOTIFY_DAYS], [""] * 6)
        self.assertEqual(notices[start + DRIFT_RENOTIFY_DAYS], "periodic")
        self.assertEqual(notices[start + DRIFT_RENOTIFY_DAYS + 1], "")

    def test_large_worsening_renotifies(self):
        self.assertEqual([self._day(4) for _ in range(DRIFT_WARN_THRESHOLD)][-1], "start")
        self.assertEqual(self._day(5), "")  # 20% → 25% は 1.5 倍に届かない
        self.assertEqual(self._day(6), "worse")  # 20% → 30%
        self.assertEqual(self._day(6), "")

    def test_resolved_is_notified_once(self):
        for _ in range(DRIFT_WARN_THRESHOLD):
            self._day(4)
        self.assertEqual(self._day(1), "resolved")
        self.assertEqual(self._day(1), "")
        # 解けた後にまた始まれば、始まった日として知らせる
        notices = [self._day(4) for _ in range(DRIFT_WARN_THRESHOLD)]
        self.assertEqual(notices[-1], "start")


class TestDecideDriftNotice(unittest.TestCase):
    def test_below_threshold_without_notice_is_silent(self):
        self.assertEqual(decide_drift_notice(DRIFT_WARN_THRESHOLD - 1, 0.2, 0, 0.0), "")
        self.assertEqual(decide_drift_notice(0, 0.0, 0, 0.0), "")

    def test_streak_restarted_counts_as_start(self):
        self.assertEqual(decide_drift_notice(DRIFT_WARN_THRESHOLD, 0.2, 9, 0.2), "start")


if __name__ == "__main__":
    unittest.main()
