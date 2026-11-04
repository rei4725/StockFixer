"""本番の組み立て _build_queue_manager が、落ちたジョブを Discord へ知らせること（#1535）

自分の notifier を渡すテストでは、run_scheduler.py の配線の抜けが見えないため、
ここでは _build_queue_manager が作った manager を動かし、送り口の手前で受ける。
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import run_scheduler  # noqa: E402

_JOB_ID = "flaky_job"
_ERROR = "接続プールのタイムアウト"


def _job_config(func, max_retries: int = 0) -> dict:
    return {
        "func": func,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-sun",
        "hour": 7,
        "minute": 30,
        "max_retries": max_retries,
        "retry_base_wait_seconds": 1.0,
    }


def _boom() -> None:
    raise RuntimeError(_ERROR)


@pytest.fixture
def results_dir(tmp_path):
    """状態ファイルを tmp_path へ向け、外への ping と待ちを止める"""
    with (
        patch("src.orchestration.scheduler_queue.get_results_dir", return_value=str(tmp_path)),
        patch("src.orchestration.scheduler_queue.ping_heartbeat"),
        patch("src.orchestration.scheduler_queue.time.sleep"),
    ):
        yield tmp_path


@pytest.fixture
def mock_send():
    with patch("src.reporting.discord.discord_utils.send_webhook_text") as send:
        yield send


def _build(config: dict):
    with patch.object(run_scheduler, "SCHEDULE_CONFIG", config):
        return run_scheduler._build_queue_manager()


def _events(results_dir) -> list[dict]:
    state = json.loads((results_dir / "scheduler_queue_state.json").read_text(encoding="utf-8"))
    return state["events"]


def test_failed_job_sends_job_id_and_error_once(results_dir, mock_send):
    """やり直しが尽きて落ちたジョブは、job_id と例外の文で1回だけ送る"""
    func = MagicMock(side_effect=_boom)
    manager = _build({_JOB_ID: _job_config(func, max_retries=2)})

    with pytest.raises(RuntimeError, match=_ERROR):
        manager.run_job(_JOB_ID, reason="test")

    assert func.call_count == 3
    mock_send.assert_called_once()
    text = mock_send.call_args.args[0]
    assert _JOB_ID in text
    assert _ERROR in text


def test_long_error_is_cut_like_stage_failure(results_dir, mock_send):
    """長い例外の文は _notify_stage_failure と同じく 1000 字で切る"""

    def _long_boom() -> None:
        raise RuntimeError("x" * 5000)

    manager = _build({_JOB_ID: _job_config(_long_boom)})

    with pytest.raises(RuntimeError):
        manager.run_job(_JOB_ID, reason="test")

    assert mock_send.call_args.args[0].count("x") == 1000


@pytest.mark.parametrize("job_id", ["daily_auto_order", "daily_rule_signals"])
def test_no_retry_job_sends_once_on_failure(results_dir, mock_send, job_id):
    """本番の設定で max_retries: 0 のジョブも、落ちた回に1回送る"""
    production_config = run_scheduler.SCHEDULE_CONFIG[job_id]
    assert production_config["max_retries"] == 0
    func = MagicMock(side_effect=_boom)
    manager = _build({job_id: {**production_config, "func": func}})

    with pytest.raises(RuntimeError, match=_ERROR):
        manager.run_job(job_id, reason="test")

    func.assert_called_once()
    mock_send.assert_called_once()
    assert job_id in mock_send.call_args.args[0]


def test_successful_job_does_not_send(results_dir, mock_send):
    manager = _build({_JOB_ID: _job_config(MagicMock())})

    assert manager.run_job(_JOB_ID, reason="test") is True

    mock_send.assert_not_called()


def test_job_recovered_by_retry_does_not_send(results_dir, mock_send):
    """やり直しで成功した回は失敗ではないので送らない"""
    func = MagicMock(side_effect=[RuntimeError(_ERROR), None])
    manager = _build({_JOB_ID: _job_config(func, max_retries=2)})

    assert manager.run_job(_JOB_ID, reason="test") is True

    mock_send.assert_not_called()


def _assert_job_error_kept(manager, results_dir) -> None:
    """ジョブの元の例外が上がり、error の記録に元の文が残ること"""
    with pytest.raises(RuntimeError, match=_ERROR):
        manager.run_job(_JOB_ID, reason="test")

    last_event = _events(results_dir)[-1]
    assert last_event["status"] == "error"
    assert last_event["error"] == _ERROR


def test_send_failure_keeps_job_error_and_record(results_dir):
    """Discord の送り口が例外を投げても、ジョブの元の例外と error の記録は変わらない"""
    manager = _build({_JOB_ID: _job_config(_boom)})

    with patch(
        "src.reporting.discord.discord_utils.send_webhook_text",
        side_effect=ValueError("Discord に届かない"),
    ) as send:
        _assert_job_error_kept(manager, results_dir)

    send.assert_called_once()


def test_notifier_failure_keeps_job_error_and_record(results_dir):
    """渡した送る関数そのものが例外を投げても、ジョブの元の例外と error の記録は変わらない"""
    with patch(
        "src.orchestration.jobs.common.notify_job_failure",
        side_effect=ValueError("送る関数が落ちた"),
    ) as notifier:
        manager = _build({_JOB_ID: _job_config(_boom)})
        _assert_job_error_kept(manager, results_dir)

    notifier.assert_called_once_with(_JOB_ID, _ERROR)
