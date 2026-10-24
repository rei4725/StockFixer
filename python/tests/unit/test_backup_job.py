"""run_daily_backup（jobs/backup_job.py）の単体テスト。"""

from unittest.mock import patch

import pytest

from src.orchestration.jobs import backup_job, daily


def _result(error=None):
    return {
        "backup_path": "data/backups/20261024_000000",
        "size_mb": 1.5,
        "elapsed_seconds": 2.0,
        "pruned_count": 1,
        "error": error,
    }


def test_daily_re_exports_run_daily_backup():
    assert daily.run_daily_backup is backup_job.run_daily_backup


def test_run_daily_backup_notifies_on_success():
    with (
        patch("src.orchestration.backup_pipeline.run_db_backup", return_value=_result()),
        patch("src.reporting.discord.discord_utils.send_backup_completion") as mock_send,
    ):
        backup_job.run_daily_backup()

    mock_send.assert_called_once()
    assert mock_send.call_args.kwargs["error"] is None


def test_run_daily_backup_raises_after_notifying_on_error():
    with (
        patch("src.orchestration.backup_pipeline.run_db_backup", return_value=_result("boom")),
        patch("src.reporting.discord.discord_utils.send_backup_completion") as mock_send,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            backup_job.run_daily_backup()

    mock_send.assert_called_once()


def test_run_daily_backup_continues_when_notification_fails():
    with (
        patch("src.orchestration.backup_pipeline.run_db_backup", return_value=_result()),
        patch(
            "src.reporting.discord.discord_utils.send_backup_completion",
            side_effect=RuntimeError("discord down"),
        ),
    ):
        backup_job.run_daily_backup()
