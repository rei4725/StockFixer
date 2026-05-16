"""ユニットテスト: 条件付きアラートサービス（NF-303）"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.utils.alert_service import (
    DRIFT_WARN_THRESHOLD,
    HEALTH_DEGRADED_THRESHOLD,
    LOSS_LIMIT_THRESHOLD,
    PIPELINE_FAIL_THRESHOLD,
    AlertResult,
    _count_consecutive_pipeline_failures,
    check_drift_warn_rule,
    check_health_degraded_rule,
    check_loss_limit_rule,
    check_pipeline_fail_rule,
    evaluate_alert_conditions,
    run_conditional_notification,
    update_drift_warn_streak,
    update_health_degraded_streak,
    update_loss_limit_streak,
)

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _make_state(events: list[dict]) -> dict:
    return {"events": events}


def _pipeline_event(status: str, finished_at: str = "2026-05-01T10:00:00+00:00") -> dict:
    return {
        "job_id": "daily_pipeline",
        "status": status,
        "finished_at": finished_at,
    }


def _write_state(path: str, events: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_make_state(events), fh)


# ---------------------------------------------------------------------------
# ルール 1: 日次パイプライン連続失敗
# ---------------------------------------------------------------------------


class TestCountConsecutivePipelineFailures(unittest.TestCase):
    def test_no_state_file_returns_zero(self):
        self.assertEqual(_count_consecutive_pipeline_failures("/nonexistent/path.json"), 0)

    def test_empty_events_returns_zero(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as fh:
            json.dump({"events": []}, fh)
            path = fh.name
        try:
            self.assertEqual(_count_consecutive_pipeline_failures(path), 0)
        finally:
            os.unlink(path)

    def test_consecutive_errors_counted(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as fh:
            fh.write("x")
            path = fh.name

        events = [
            _pipeline_event("error", "2026-05-03T10:00:00+00:00"),
            _pipeline_event("error", "2026-05-02T10:00:00+00:00"),
            _pipeline_event("success", "2026-05-01T10:00:00+00:00"),
        ]
        _write_state(path, events)
        try:
            self.assertEqual(_count_consecutive_pipeline_failures(path), 2)
        finally:
            os.unlink(path)

    def test_success_breaks_streak(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as fh:
            fh.write("x")
            path = fh.name

        events = [
            _pipeline_event("error", "2026-05-04T10:00:00+00:00"),
            _pipeline_event("success", "2026-05-03T10:00:00+00:00"),
            _pipeline_event("error", "2026-05-02T10:00:00+00:00"),
        ]
        _write_state(path, events)
        try:
            self.assertEqual(_count_consecutive_pipeline_failures(path), 1)
        finally:
            os.unlink(path)

    def test_ignores_other_job_ids(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as fh:
            fh.write("x")
            path = fh.name

        events = [
            {
                "job_id": "weekly_training",
                "status": "error",
                "finished_at": "2026-05-04T10:00:00+00:00",
            },
            _pipeline_event("success", "2026-05-03T10:00:00+00:00"),
        ]
        _write_state(path, events)
        try:
            self.assertEqual(_count_consecutive_pipeline_failures(path), 0)
        finally:
            os.unlink(path)


class TestCheckPipelineFailRule(unittest.TestCase):
    def _run_with_count(self, count: int) -> AlertResult:
        with patch(
            "src.services.alert_service._count_consecutive_pipeline_failures",
            return_value=count,
        ):
            return check_pipeline_fail_rule()

    def test_below_threshold_not_triggered(self):
        result = self._run_with_count(PIPELINE_FAIL_THRESHOLD - 1)
        self.assertFalse(result.triggered)
        self.assertEqual(result.rule_id, "NF-303-1")

    def test_at_threshold_triggered(self):
        result = self._run_with_count(PIPELINE_FAIL_THRESHOLD)
        self.assertTrue(result.triggered)

    def test_above_threshold_triggered(self):
        result = self._run_with_count(PIPELINE_FAIL_THRESHOLD + 1)
        self.assertTrue(result.triggered)


# ---------------------------------------------------------------------------
# ルール 2: 損失上限連続発動
# ---------------------------------------------------------------------------


class TestLossLimitStreak(unittest.TestCase):
    def _mock_config(self, current_value: int):
        return patch(
            "src.services.alert_service._get_streak",
            return_value=current_value,
        )

    def test_update_increments_when_triggered(self):
        with (
            patch("src.services.alert_service._get_streak", return_value=1),
            patch("src.services.alert_service._set_streak") as mock_set,
        ):
            result = update_loss_limit_streak(True)
        self.assertEqual(result, 2)
        mock_set.assert_called_once()

    def test_update_resets_when_not_triggered(self):
        with (
            patch("src.services.alert_service._get_streak", return_value=2),
            patch("src.services.alert_service._set_streak") as mock_set,
        ):
            result = update_loss_limit_streak(False)
        self.assertEqual(result, 0)
        mock_set.assert_called_once()

    def test_check_below_threshold_not_triggered(self):
        with patch("src.services.alert_service._get_streak", return_value=LOSS_LIMIT_THRESHOLD - 1):
            result = check_loss_limit_rule()
        self.assertFalse(result.triggered)
        self.assertEqual(result.rule_id, "NF-303-2")

    def test_check_at_threshold_triggered(self):
        with patch("src.services.alert_service._get_streak", return_value=LOSS_LIMIT_THRESHOLD):
            result = check_loss_limit_rule()
        self.assertTrue(result.triggered)


# ---------------------------------------------------------------------------
# ルール 3: ドリフト連続警告
# ---------------------------------------------------------------------------


class TestDriftWarnStreak(unittest.TestCase):
    def test_update_increments_when_warn(self):
        with (
            patch("src.services.alert_service._get_streak", return_value=0),
            patch("src.services.alert_service._set_streak") as mock_set,
        ):
            result = update_drift_warn_streak(True)
        self.assertEqual(result, 1)
        mock_set.assert_called_once()

    def test_update_resets_when_no_warn(self):
        with (
            patch("src.services.alert_service._get_streak", return_value=3),
            patch("src.services.alert_service._set_streak") as mock_set,
        ):
            result = update_drift_warn_streak(False)
        self.assertEqual(result, 0)
        mock_set.assert_called_once()

    def test_check_below_threshold_not_triggered(self):
        with patch("src.services.alert_service._get_streak", return_value=DRIFT_WARN_THRESHOLD - 1):
            result = check_drift_warn_rule()
        self.assertFalse(result.triggered)
        self.assertEqual(result.rule_id, "NF-303-3")

    def test_check_at_threshold_triggered(self):
        with patch("src.services.alert_service._get_streak", return_value=DRIFT_WARN_THRESHOLD):
            result = check_drift_warn_rule()
        self.assertTrue(result.triggered)


# ---------------------------------------------------------------------------
# ルール 4: /health degraded 継続
# ---------------------------------------------------------------------------


class TestHealthDegradedStreak(unittest.TestCase):
    def test_update_increments_when_degraded(self):
        with (
            patch("src.services.alert_service._get_streak", return_value=1),
            patch("src.services.alert_service._set_streak") as mock_set,
        ):
            result = update_health_degraded_streak(True)
        self.assertEqual(result, 2)
        mock_set.assert_called_once()

    def test_update_resets_when_ok(self):
        with (
            patch("src.services.alert_service._get_streak", return_value=5),
            patch("src.services.alert_service._set_streak") as mock_set,
        ):
            result = update_health_degraded_streak(False)
        self.assertEqual(result, 0)
        mock_set.assert_called_once()

    def test_check_below_threshold_not_triggered(self):
        with patch(
            "src.services.alert_service._get_streak", return_value=HEALTH_DEGRADED_THRESHOLD - 1
        ):
            result = check_health_degraded_rule()
        self.assertFalse(result.triggered)
        self.assertEqual(result.rule_id, "NF-303-4")

    def test_check_at_threshold_triggered(self):
        with patch(
            "src.services.alert_service._get_streak", return_value=HEALTH_DEGRADED_THRESHOLD
        ):
            result = check_health_degraded_rule()
        self.assertTrue(result.triggered)


# ---------------------------------------------------------------------------
# evaluate_alert_conditions
# ---------------------------------------------------------------------------


class TestEvaluateAlertConditions(unittest.TestCase):
    def _all_ok_results(self) -> list[AlertResult]:
        return [
            AlertResult("NF-303-1", "A", False, 0, 2),
            AlertResult("NF-303-2", "B", False, 0, 3),
            AlertResult("NF-303-3", "C", False, 0, 2),
            AlertResult("NF-303-4", "D", False, 0, 2),
        ]

    def test_returns_four_results(self):
        with (
            patch(
                "src.services.alert_service.check_pipeline_fail_rule",
                return_value=AlertResult("NF-303-1", "A", False, 0, 2),
            ),
            patch(
                "src.services.alert_service.check_loss_limit_rule",
                return_value=AlertResult("NF-303-2", "B", False, 0, 3),
            ),
            patch(
                "src.services.alert_service.check_drift_warn_rule",
                return_value=AlertResult("NF-303-3", "C", False, 0, 2),
            ),
            patch(
                "src.services.alert_service.check_health_degraded_rule",
                return_value=AlertResult("NF-303-4", "D", False, 0, 2),
            ),
        ):
            results = evaluate_alert_conditions()
        self.assertEqual(len(results), 4)

    def test_all_ok_no_triggered(self):
        with (
            patch(
                "src.services.alert_service.check_pipeline_fail_rule",
                return_value=AlertResult("NF-303-1", "A", False, 0, 2),
            ),
            patch(
                "src.services.alert_service.check_loss_limit_rule",
                return_value=AlertResult("NF-303-2", "B", False, 0, 3),
            ),
            patch(
                "src.services.alert_service.check_drift_warn_rule",
                return_value=AlertResult("NF-303-3", "C", False, 0, 2),
            ),
            patch(
                "src.services.alert_service.check_health_degraded_rule",
                return_value=AlertResult("NF-303-4", "D", False, 0, 2),
            ),
        ):
            results = evaluate_alert_conditions()
        self.assertFalse(any(r.triggered for r in results))

    def test_one_triggered(self):
        with (
            patch(
                "src.services.alert_service.check_pipeline_fail_rule",
                return_value=AlertResult("NF-303-1", "A", True, 3, 2),
            ),
            patch(
                "src.services.alert_service.check_loss_limit_rule",
                return_value=AlertResult("NF-303-2", "B", False, 0, 3),
            ),
            patch(
                "src.services.alert_service.check_drift_warn_rule",
                return_value=AlertResult("NF-303-3", "C", False, 0, 2),
            ),
            patch(
                "src.services.alert_service.check_health_degraded_rule",
                return_value=AlertResult("NF-303-4", "D", False, 0, 2),
            ),
        ):
            results = evaluate_alert_conditions()
        triggered = [r for r in results if r.triggered]
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0].rule_id, "NF-303-1")


# ---------------------------------------------------------------------------
# run_conditional_notification
# ---------------------------------------------------------------------------


class TestRunConditionalNotification(unittest.TestCase):
    def test_sends_detail_when_triggered(self):
        results = [
            AlertResult("NF-303-1", "パイプライン失敗", True, 3, 2),
            AlertResult("NF-303-2", "損失上限", False, 0, 3),
            AlertResult("NF-303-3", "ドリフト", False, 0, 2),
            AlertResult("NF-303-4", "ヘルス", False, 0, 2),
        ]
        with patch(
            "src.services.alert_service._send_alert_detail", return_value=True
        ) as mock_detail, patch("src.services.alert_service._send_daily_summary") as mock_summary:
            ok = run_conditional_notification(results=results)

        mock_detail.assert_called_once_with(results)
        mock_summary.assert_not_called()
        self.assertTrue(ok)

    def test_sends_summary_when_all_ok(self):
        results = [
            AlertResult("NF-303-1", "A", False, 0, 2),
            AlertResult("NF-303-2", "B", False, 0, 3),
            AlertResult("NF-303-3", "C", False, 0, 2),
            AlertResult("NF-303-4", "D", False, 0, 2),
        ]
        with patch("src.services.alert_service._send_alert_detail") as mock_detail, patch(
            "src.services.alert_service._send_daily_summary", return_value=True
        ) as mock_summary:
            ok = run_conditional_notification(results=results)

        mock_summary.assert_called_once_with(results)
        mock_detail.assert_not_called()
        self.assertTrue(ok)

    def test_evaluates_conditions_when_results_none(self):
        all_ok = [
            AlertResult("NF-303-1", "A", False, 0, 2),
            AlertResult("NF-303-2", "B", False, 0, 3),
            AlertResult("NF-303-3", "C", False, 0, 2),
            AlertResult("NF-303-4", "D", False, 0, 2),
        ]
        with patch(
            "src.services.alert_service.evaluate_alert_conditions", return_value=all_ok
        ) as mock_eval, patch("src.services.alert_service._send_daily_summary", return_value=True):
            run_conditional_notification(results=None)
        mock_eval.assert_called_once()


# ---------------------------------------------------------------------------
# AlertResult.as_discord_lines
# ---------------------------------------------------------------------------


class TestAlertResultDiscordLines(unittest.TestCase):
    def test_triggered_shows_alert(self):
        r = AlertResult("NF-303-1", "テスト", True, 3, 2)
        lines = r.as_discord_lines()
        self.assertTrue(any("アラート" in line for line in lines))

    def test_not_triggered_shows_ok(self):
        r = AlertResult("NF-303-1", "テスト", False, 1, 2)
        lines = r.as_discord_lines()
        self.assertTrue(any("✅" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
