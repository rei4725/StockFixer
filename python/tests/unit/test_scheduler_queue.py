import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.orchestration.scheduler_queue import SchedulerQueueManager  # noqa: E402


def _build_config(counter: dict[str, int]):
    def _daily_func():
        counter["daily"] = counter.get("daily", 0) + 1

    def _weekly_func():
        counter["weekly"] = counter.get("weekly", 0) + 1

    return {
        "daily_pipeline": {
            "func": _daily_func,
            "period": "daily",
            "day_of_week": "mon-fri",
            "hour": 19,
            "minute": 0,
            "recovery_delay_minutes": 10,
            "max_executions_per_period": 2,
            "backfill": True,
        },
        "weekly_model_training": {
            "func": _weekly_func,
            "period": "weekly",
            "day_of_week": "sat",
            "hour": 3,
            "minute": 0,
            "recovery_delay_minutes": 15,
            "max_executions_per_period": 2,
        },
    }


def test_should_recover_true_when_schedule_passed_and_no_success(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 2026-03-02 は月曜
    now = datetime(2026, 3, 2, 19, 20, tzinfo=manager.tz)

    assert manager.should_recover("daily_pipeline", now=now) is True


def test_should_recover_false_after_success_event(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 2026-03-02 分として実行記録（period_key を明示して時刻のズレを回避）
    manager.run_job("daily_pipeline", reason="scheduled", period_key="2026-03-02")

    now = datetime(2026, 3, 2, 19, 20, tzinfo=manager.tz)
    assert manager.should_recover("daily_pipeline", now=now) is False


def test_should_recover_false_for_cron_string_hour_instead_of_crashing(tmp_path):
    """hour が "*/2" のようなcron式（1日複数回実行のジョブ）の場合、
    datetime.replace() に渡せずクラッシュしていた（recovery_pollerが5分毎に
    異常終了するバグ）。「所定の1時刻」の前提が成り立たないジョブは
    補完ポーリング対象外としてFalseを返すのが正しい挙動。
    """
    counter = {}

    def _promotion_check_func():
        counter["promotion_check"] = counter.get("promotion_check", 0) + 1

    config = {
        "strategy_promotion_check": {
            "func": _promotion_check_func,
            "period": "daily",
            "day_of_week": "mon-sun",
            "hour": "*/2",
            "minute": 30,
            "recovery_delay_minutes": 30,
            "max_executions_per_period": 12,
        },
    }
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    now = datetime(2026, 3, 2, 19, 20, tzinfo=manager.tz)
    assert manager.should_recover("strategy_promotion_check", now=now) is False


def _build_production_like_daily_config(func):
    """本番と同じ 07:30 JST（22:30Z）・上限3回の日次ジョブ設定"""
    return {
        "daily_pipeline": {
            "func": func,
            "period": "daily",
            "day_of_week": "mon-fri",
            "hour": 7,
            "minute": 30,
            "recovery_delay_minutes": 10,
            "max_executions_per_period": 3,
        }
    }


def test_should_recover_false_while_scheduled_run_is_in_progress(tmp_path):
    """22:30Z の回が動いている間は、補完ポーリングが補完を始めないこと（#806）。
    以前は should_recover が True を返し、5分おきに「補完実行を開始」の行が出ていた。
    """
    import threading

    started = threading.Event()
    released = threading.Event()

    def slow_func():
        started.set()
        released.wait()

    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(
        _build_production_like_daily_config(slow_func), state_file_path=str(state_path)
    )
    # 2026-10-19 は月曜。07:30 JST（2026-10-18 22:30Z）の定時の回
    scheduled_now = datetime(2026, 10, 19, 7, 30, tzinfo=manager.tz)
    manager.now = lambda: scheduled_now  # type: ignore[method-assign]

    t = threading.Thread(target=lambda: manager.run_job("daily_pipeline", reason="scheduled"))
    t.start()
    try:
        assert started.wait(timeout=5)
        # 07:40〜08:10 JST（22:40Z〜23:10Z）の補完ポーリング
        for minute_offset in range(0, 35, 5):
            poll_now = datetime(2026, 10, 19, 7, 40, tzinfo=manager.tz) + timedelta(
                minutes=minute_offset
            )
            assert manager.should_recover("daily_pipeline", now=poll_now) is False
    finally:
        released.set()
        t.join()


def test_should_recover_false_after_scheduled_run_completes_same_day(tmp_path):
    """22:30Z の回が完了した日は、その日の補完が始まらないこと（#806）。
    完了の記録の period_key と should_recover の period_key が同じ JST の日付になる。
    """
    counter = {"n": 0}

    def _func():
        counter["n"] += 1

    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(
        _build_production_like_daily_config(_func), state_file_path=str(state_path)
    )
    scheduled_now = datetime(2026, 10, 19, 7, 30, tzinfo=manager.tz)
    manager.now = lambda: scheduled_now  # type: ignore[method-assign]

    assert manager.run_job("daily_pipeline", reason="scheduled") is True
    assert manager.has_success("daily_pipeline", "2026-10-19") is True

    for poll_now in (
        datetime(2026, 10, 19, 8, 15, tzinfo=manager.tz),
        datetime(2026, 10, 19, 23, 55, tzinfo=manager.tz),
    ):
        assert manager.should_recover("daily_pipeline", now=poll_now) is False
    # 再起動の後も同じ
    reloaded = SchedulerQueueManager(
        _build_production_like_daily_config(_func), state_file_path=str(state_path)
    )
    assert (
        reloaded.should_recover(
            "daily_pipeline", now=datetime(2026, 10, 19, 8, 15, tzinfo=reloaded.tz)
        )
        is False
    )
    assert counter["n"] == 1


def test_run_job_respects_execution_limit(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    assert manager.run_job("daily_pipeline", reason="scheduled") is True
    assert manager.run_job("daily_pipeline", reason="recovery") is True
    assert manager.run_job("daily_pipeline", reason="recovery") is False

    assert counter["daily"] == 2


def test_recovery_stops_at_limit_when_process_is_killed_mid_job(tmp_path):
    """ジョブの途中でプロセスが消えた（finally が走らない）実行も回数に数え、
    同じ period_key の補完が max_executions_per_period 回で止まること。
    """
    state_path = tmp_path / "scheduler_queue_state.json"
    calls = {"n": 0}
    snapshot: dict[str, bytes] = {}

    def _killed_func():
        calls["n"] += 1
        # プロセスが消えた時点でディスクに残っている状態を写し取る
        snapshot["state"] = state_path.read_bytes() if state_path.exists() else b""

    config = {
        "weekly_model_training": {
            "func": _killed_func,
            "period": "weekly",
            "day_of_week": "sat",
            "hour": 3,
            "minute": 0,
            "recovery_delay_minutes": 15,
            "max_executions_per_period": 2,
            "max_retries": 0,
        }
    }

    # 2026-03-07 は土曜
    tz = SchedulerQueueManager(config, state_file_path=str(state_path)).tz
    now = datetime(2026, 3, 7, 3, 20, tzinfo=tz)

    for _ in range(10):
        # 再起動したプロセスとして状態ファイルを読み直す
        manager = SchedulerQueueManager(config, state_file_path=str(state_path))
        manager.now = lambda: now  # type: ignore[method-assign]
        if not manager.should_recover("weekly_model_training", now=now):
            break
        manager.run_job("weekly_model_training", reason="recovery")
        # finally が書いた記録を捨て、消えた時点の状態に戻す
        if snapshot["state"]:
            state_path.write_bytes(snapshot["state"])
        else:
            state_path.unlink()

    assert calls["n"] == 2
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    assert manager.has_success("weekly_model_training", "2026-W10") is False
    assert manager.get_execution_count("weekly_model_training", "2026-W10") == 2


def test_execution_count_reads_legacy_state_without_started_events(tmp_path):
    """started の記録が無い前の版の状態ファイルでも、success・error を数えること"""
    state_path = tmp_path / "scheduler_queue_state.json"
    legacy_events = [
        {"job_id": "daily_pipeline", "period_key": "2026-03-02", "status": "error"},
        {"job_id": "daily_pipeline", "period_key": "2026-03-02", "status": "retry"},
        {"job_id": "daily_pipeline", "period_key": "2026-03-02", "status": "success"},
        {"job_id": "daily_pipeline", "period_key": "2026-03-02", "status": "skipped_limit"},
    ]
    state_path.write_text(json.dumps({"events": legacy_events}), encoding="utf-8")
    manager = SchedulerQueueManager(_build_config({}), state_file_path=str(state_path))

    assert manager.get_execution_count("daily_pipeline", "2026-03-02") == 2


def test_finished_run_is_counted_once(tmp_path):
    """始めの記録と終わりの記録がそろった実行は1回に数えること"""
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    manager.run_job("daily_pipeline", reason="scheduled", period_key="2026-03-02")

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    statuses = [e["status"] for e in saved["events"]]
    assert statuses == ["started", "success"]
    assert manager.get_execution_count("daily_pipeline", "2026-03-02") == 1


def test_run_job_force_bypasses_execution_limit(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    assert manager.run_job("daily_pipeline", reason="scheduled") is True
    assert manager.run_job("daily_pipeline", reason="recovery") is True
    assert manager.run_job("daily_pipeline", reason="manual", force=True) is True

    assert counter["daily"] == 3


def test_get_missed_past_periods_detects_missed_day(tmp_path):
    """前日分が未実行の場合に get_missed_past_periods が検出する"""
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 再起動後の早朝（当日 19:00 前）を模倣
    # 2026-03-03 (火) 08:00 → 前日 2026-03-02 (月) 分が未実行
    now = datetime(2026, 3, 3, 8, 0, tzinfo=manager.tz)
    missed = manager.get_missed_past_periods("daily_pipeline", lookback_days=2, now=now)
    assert "2026-03-02" in missed


def test_get_missed_past_periods_empty_when_already_run(tmp_path):
    """前日分が成功済みの場合は空リストを返す"""
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 前日 (2026-03-02) の実行を記録
    manager.run_job("daily_pipeline", reason="scheduled", period_key="2026-03-02")

    now = datetime(2026, 3, 3, 8, 0, tzinfo=manager.tz)
    missed = manager.get_missed_past_periods("daily_pipeline", lookback_days=2, now=now)
    assert "2026-03-02" not in missed


def test_get_missed_past_periods_skips_non_scheduled_days(tmp_path):
    """週末は対象外のためスキップされる"""
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    # 2026-03-09 (月) 08:00 → 前日 2026-03-08 (日) は mon-fri 対象外
    now = datetime(2026, 3, 9, 8, 0, tzinfo=manager.tz)
    missed = manager.get_missed_past_periods("daily_pipeline", lookback_days=1, now=now)
    assert "2026-03-08" not in missed


@pytest.mark.parametrize(
    "job_id", ["daily_auto_order", "daily_settle_orders", "daily_rule_signals"]
)
def test_get_missed_past_periods_skips_order_jobs_in_schedule_config(tmp_path, job_id):
    """発注・約定のジョブは前日分の鍵で動かさない（同じ営業日の二重発注を防ぐ。#825）"""
    from run_scheduler import SCHEDULE_CONFIG

    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(tmp_path / "state.json"))

    # 2026-03-03 (火) 10:00 → 前日 2026-03-02 (月) 分は未実行
    now = datetime(2026, 3, 3, 10, 0, tzinfo=manager.tz)
    assert manager.get_missed_past_periods(job_id, lookback_days=2, now=now) == []


def test_get_missed_past_periods_backfills_daily_pipeline_in_schedule_config(tmp_path):
    """daily_pipeline は前日分の補完を続ける"""
    from run_scheduler import SCHEDULE_CONFIG

    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(tmp_path / "state.json"))

    now = datetime(2026, 3, 3, 10, 0, tzinfo=manager.tz)
    missed = manager.get_missed_past_periods("daily_pipeline", lookback_days=2, now=now)
    assert "2026-03-02" in missed


def test_get_missed_past_periods_empty_without_backfill_flag(tmp_path):
    """backfill の印が無い daily ジョブは前日分を補完しない（既定で安全な側）"""
    config = _build_config({})
    del config["daily_pipeline"]["backfill"]
    manager = SchedulerQueueManager(config, state_file_path=str(tmp_path / "state.json"))

    now = datetime(2026, 3, 3, 8, 0, tzinfo=manager.tz)
    assert manager.get_missed_past_periods("daily_pipeline", lookback_days=2, now=now) == []


# ============================================
# 当日分の補完の締め（recovery_until。#835）
# ============================================


def test_should_recover_daily_auto_order_true_before_cutoff(tmp_path):
    """daily_auto_order の当日分の補完は、締めより前なら今のとおり動く"""
    from run_scheduler import SCHEDULE_CONFIG

    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(tmp_path / "state.json"))

    # 2026-03-02 は月曜。08:50 + 10分を過ぎ、締めより前
    now = datetime(2026, 3, 2, 9, 10, tzinfo=manager.tz)
    assert manager.should_recover("daily_auto_order", now=now) is True


@pytest.mark.parametrize("hour, minute", [(9, 30), (11, 0), (16, 0), (23, 55)])
def test_should_recover_daily_auto_order_false_after_cutoff(tmp_path, hour, minute):
    """daily_auto_order の当日分の補完は、締めを過ぎたら動かない（引けの後の発注を防ぐ）"""
    from run_scheduler import SCHEDULE_CONFIG

    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(tmp_path / "state.json"))

    now = datetime(2026, 3, 2, hour, minute, tzinfo=manager.tz)
    assert manager.should_recover("daily_auto_order", now=now) is False


@pytest.mark.parametrize("job_id", ["daily_pipeline", "daily_settle_orders", "daily_rule_signals"])
def test_should_recover_jobs_without_cutoff_still_recover_late(tmp_path, job_id):
    """締めの欄の無いジョブの当日分の補完は、夜でも今のとおり動く"""
    from run_scheduler import SCHEDULE_CONFIG

    assert "recovery_until" not in SCHEDULE_CONFIG[job_id]
    manager = SchedulerQueueManager(SCHEDULE_CONFIG, state_file_path=str(tmp_path / "state.json"))

    now = datetime(2026, 3, 2, 23, 0, tzinfo=manager.tz)
    assert manager.should_recover(job_id, now=now) is True


def test_should_recover_respects_recovery_until_in_config(tmp_path):
    """recovery_until を持つジョブは、締めの時刻ちょうどから補完しない"""
    config = _build_config({})
    config["daily_pipeline"]["recovery_until"] = (21, 0)
    manager = SchedulerQueueManager(config, state_file_path=str(tmp_path / "state.json"))

    assert manager.should_recover(
        "daily_pipeline", now=datetime(2026, 3, 2, 20, 59, tzinfo=manager.tz)
    )
    assert not manager.should_recover(
        "daily_pipeline", now=datetime(2026, 3, 2, 21, 0, tzinfo=manager.tz)
    )


# ============================================
# 月次ジョブの期間キーと補完判定
# ============================================


def _build_monthly_config():
    return {
        "monthly_report": {
            "func": lambda: None,
            "period": "monthly",
            "day_of_week": "mon-sun",
            "day": 1,
            "hour": 3,
            "minute": 0,
            "recovery_delay_minutes": 60,
            "max_executions_per_period": 1,
        },
    }


def test_build_period_key_returns_month_for_monthly(tmp_path):
    manager = SchedulerQueueManager(
        _build_monthly_config(), state_file_path=str(tmp_path / "state.json")
    )
    config = manager.schedule_config["monthly_report"]

    now = datetime(2026, 10, 15, 12, 0, tzinfo=manager.tz)
    assert manager._build_period_key(config, now) == "2026-10"


def test_should_recover_monthly_false_after_success_on_first_day(tmp_path):
    """10-01 に成功があれば、同じ月の 10-15 には補完しない"""
    manager = SchedulerQueueManager(
        _build_monthly_config(), state_file_path=str(tmp_path / "state.json")
    )
    first_day = datetime(2026, 10, 1, 3, 0, tzinfo=manager.tz)
    manager.now = lambda: first_day  # type: ignore[method-assign]
    manager.run_job("monthly_report", reason="scheduled")

    now = datetime(2026, 10, 15, 12, 0, tzinfo=manager.tz)
    assert manager.should_recover("monthly_report", now=now) is False


def test_should_recover_monthly_true_when_first_day_missed(tmp_path):
    """10-01 に成功が無ければ、10-02 に補完する"""
    manager = SchedulerQueueManager(
        _build_monthly_config(), state_file_path=str(tmp_path / "state.json")
    )

    now = datetime(2026, 10, 2, 8, 0, tzinfo=manager.tz)
    assert manager.should_recover("monthly_report", now=now) is True


def test_should_recover_monthly_false_before_scheduled_time_on_first_day(tmp_path):
    """11-01 の所定の時刻の前は、前の月の鍵でも新しい月の鍵でも補完しない"""
    manager = SchedulerQueueManager(
        _build_monthly_config(), state_file_path=str(tmp_path / "state.json")
    )

    now = datetime(2026, 11, 1, 3, 30, tzinfo=manager.tz)
    assert manager.should_recover("monthly_report", now=now) is False


def test_should_recover_monthly_new_month_ignores_previous_month_success(tmp_path):
    """前の月の成功は新しい月の判定に数えない（月の鍵で区切る）"""
    manager = SchedulerQueueManager(
        _build_monthly_config(), state_file_path=str(tmp_path / "state.json")
    )
    first_day = datetime(2026, 10, 1, 3, 0, tzinfo=manager.tz)
    manager.now = lambda: first_day  # type: ignore[method-assign]
    manager.run_job("monthly_report", reason="scheduled")
    assert manager.has_success("monthly_report", "2026-10") is True

    assert (
        manager.should_recover(
            "monthly_report", now=datetime(2026, 10, 31, 12, 0, tzinfo=manager.tz)
        )
        is False
    )
    assert (
        manager.should_recover(
            "monthly_report", now=datetime(2026, 11, 1, 4, 30, tzinfo=manager.tz)
        )
        is True
    )


def test_should_recover_monthly_false_before_configured_day(tmp_path):
    """設定の day より前の日は補完しない"""
    config = _build_monthly_config()
    config["monthly_report"]["day"] = 15
    manager = SchedulerQueueManager(config, state_file_path=str(tmp_path / "state.json"))

    assert (
        manager.should_recover(
            "monthly_report", now=datetime(2026, 10, 14, 12, 0, tzinfo=manager.tz)
        )
        is False
    )
    assert (
        manager.should_recover(
            "monthly_report", now=datetime(2026, 10, 15, 4, 30, tzinfo=manager.tz)
        )
        is True
    )


def test_get_missed_past_periods_empty_for_monthly(tmp_path):
    """monthly の取りこぼしは should_recover が月の鍵で拾うため、過去日の探索は空を返す"""
    manager = SchedulerQueueManager(
        _build_monthly_config(), state_file_path=str(tmp_path / "state.json")
    )

    now = datetime(2026, 10, 2, 1, 0, tzinfo=manager.tz)
    assert manager.get_missed_past_periods("monthly_report", lookback_days=2, now=now) == []


def test_run_job_persists_timestamps_in_utc(tmp_path):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    assert manager.run_job("daily_pipeline", reason="scheduled") is True

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    event = saved["events"][-1]
    assert event["started_at"].endswith("+00:00")
    assert event["finished_at"].endswith("+00:00")


def test_run_job_logs_start_and_finish_with_duration(tmp_path, caplog):
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    with caplog.at_level("INFO", logger="scheduler"):
        assert manager.run_job("daily_pipeline", reason="scheduled") is True

    payloads = [
        json.loads(record.getMessage()) for record in caplog.records if record.name == "scheduler"
    ]

    assert any(
        p.get("event") == "scheduler_job"
        and p.get("phase") == "start"
        and p.get("job_id") == "daily_pipeline"
        for p in payloads
    )
    assert any(
        p.get("event") == "scheduler_job"
        and p.get("phase") == "finish"
        and p.get("job_id") == "daily_pipeline"
        and p.get("status") == "success"
        and "duration_seconds" in p
        for p in payloads
    )


def test_run_job_logs_failure_with_duration(tmp_path, caplog):
    def _failing_func():
        raise RuntimeError("intentional failure")

    config = {
        "daily_pipeline": {
            "func": _failing_func,
            "period": "daily",
            "day_of_week": "mon-fri",
            "hour": 19,
            "minute": 0,
            "recovery_delay_minutes": 10,
            "max_executions_per_period": 1,
            "max_retries": 0,
        }
    }

    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    with caplog.at_level("INFO", logger="scheduler"):
        with pytest.raises(RuntimeError, match="intentional failure"):
            manager.run_job("daily_pipeline", reason="manual", force=True)

    payloads = [
        json.loads(record.getMessage()) for record in caplog.records if record.name == "scheduler"
    ]

    assert any(
        p.get("event") == "scheduler_job"
        and p.get("phase") == "start"
        and p.get("job_id") == "daily_pipeline"
        for p in payloads
    )
    assert any(
        p.get("event") == "scheduler_job"
        and p.get("phase") == "finish"
        and p.get("job_id") == "daily_pipeline"
        and p.get("status") == "error"
        and p.get("error") == "intentional failure"
        and "duration_seconds" in p
        for p in payloads
    )


def test_in_progress_job_is_skipped(tmp_path):
    """実行中のジョブは2回目の run_job でスキップされること"""
    state_path = tmp_path / "state.json"
    import threading

    barrier = threading.Barrier(2)
    released = threading.Event()

    def slow_func():
        barrier.wait()  # テストスレッドと同期
        released.wait()  # 解放まで待機

    config = {
        "slow_job": {
            "func": slow_func,
            "period": "daily",
            "day_of_week": "*",
            "hour": 8,
            "minute": 0,
            "max_executions_per_period": 5,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    # 別スレッドでジョブを実行開始
    t = threading.Thread(target=lambda: manager.run_job("slow_job", "background"))
    t.start()
    barrier.wait()  # slow_func が実行開始するまで待つ

    # この時点でジョブは _in_progress_jobs にある → スキップされるはず
    result = manager.run_job("slow_job", reason="concurrent", force=True)
    # force=True でも in_progress は優先
    assert result is False

    released.set()
    t.join()


def test_weekly_period_key(tmp_path):
    """weekly ジョブの period_key が ISO 週番号形式になること"""
    from zoneinfo import ZoneInfo

    state_path = tmp_path / "state.json"
    config = {
        "weekly_job": {
            "func": lambda: None,
            "period": "weekly",
            "day_of_week": "sat",
            "hour": 10,
            "minute": 0,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    tz = ZoneInfo("Asia/Tokyo")
    dt = datetime(2026, 1, 3, 10, 0, 0, tzinfo=tz)  # 2026-01-03 は土曜
    key = manager._build_period_key(config["weekly_job"], dt)
    assert key.startswith("2026-W")


def test_is_scheduled_day_wildcard(tmp_path):
    """day_of_week='*' のとき全日が対象になること"""
    from zoneinfo import ZoneInfo

    state_path = tmp_path / "state.json"
    config = {
        "any_day": {
            "func": lambda: None,
            "period": "daily",
            "day_of_week": "*",
            "hour": 8,
            "minute": 0,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    tz = ZoneInfo("Asia/Tokyo")
    for day in range(7):
        dt = datetime(2026, 1, 5 + day, 10, 0, 0, tzinfo=tz)
        assert manager._is_scheduled_day(config["any_day"], dt) is True


def test_load_state_with_invalid_json(tmp_path):
    """不正な JSON の場合、空イベントリストで初期化されること"""
    state_path = tmp_path / "state.json"
    state_path.write_text("not valid json", encoding="utf-8")
    config = {"job": {"func": lambda: None, "period": "daily", "hour": 8, "minute": 0}}
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    assert manager._state == {"events": []}


def test_load_state_with_non_dict_state(tmp_path):
    """JSON がリスト形式の場合も空イベントリストで初期化されること"""
    state_path = tmp_path / "state.json"
    state_path.write_text('["not", "a", "dict"]', encoding="utf-8")
    config = {"job": {"func": lambda: None, "period": "daily", "hour": 8, "minute": 0}}
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    assert manager._state == {"events": []}


# ============================================
# 複数のプロセス・スレッドからの状態ファイルの扱い（#826）
# ============================================


def _build_once_daily_config(counter: dict[str, int]):
    def _func():
        counter["n"] = counter.get("n", 0) + 1

    return {
        "daily_auto_order": {
            "func": _func,
            "period": "daily",
            "day_of_week": "mon-fri",
            "hour": 8,
            "minute": 50,
            "recovery_delay_minutes": 10,
            "max_executions_per_period": 1,
            "max_retries": 0,
            "backfill": True,
        }
    }


def test_other_manager_success_is_counted_by_run_job_and_should_recover(tmp_path):
    """手での即時実行（別の manager）の成功を、常駐の manager が数えること（#826）"""
    counter: dict[str, int] = {}
    state_path = str(tmp_path / "scheduler_queue_state.json")
    resident = SchedulerQueueManager(_build_once_daily_config(counter), state_file_path=state_path)
    manual = SchedulerQueueManager(_build_once_daily_config(counter), state_file_path=state_path)
    # 2026-10-19 は月曜
    run_now = datetime(2026, 10, 19, 8, 30, tzinfo=resident.tz)
    resident.now = lambda: run_now  # type: ignore[method-assign]
    manual.now = lambda: run_now  # type: ignore[method-assign]

    assert manual.run_job("daily_auto_order", reason="manual") is True

    poll_now = datetime(2026, 10, 19, 9, 5, tzinfo=resident.tz)
    assert resident.should_recover("daily_auto_order", now=poll_now) is False
    assert resident.run_job("daily_auto_order", reason="scheduled") is False
    assert counter["n"] == 1


def test_alternating_managers_do_not_drop_each_others_events(tmp_path):
    """2つの manager が交互に記録を足しても、どちらの記録も残ること（#826）"""
    state_file = tmp_path / "scheduler_queue_state.json"
    state_path = str(state_file)
    first = SchedulerQueueManager(_build_config({}), state_file_path=state_path)
    second = SchedulerQueueManager(_build_config({}), state_file_path=state_path)

    first.run_job("daily_pipeline", reason="scheduled", period_key="2026-03-02")
    second.run_job("daily_pipeline", reason="manual", period_key="2026-03-03")
    first.run_job("daily_pipeline", reason="scheduled", period_key="2026-03-04")

    saved = json.loads(state_file.read_text(encoding="utf-8"))
    keys = [e["period_key"] for e in saved["events"] if e["status"] == "success"]
    assert keys == ["2026-03-02", "2026-03-03", "2026-03-04"]

    reloaded = SchedulerQueueManager(_build_config({}), state_file_path=state_path)
    for key in ("2026-03-02", "2026-03-03", "2026-03-04"):
        assert reloaded.has_success("daily_pipeline", key) is True


def test_corrupted_state_file_stops_recovery_and_logs_error(tmp_path, caplog):
    """途中で切れた状態ファイルは退避し、ERROR を出して補完を止めること（#826）"""
    counter: dict[str, int] = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    state_path.write_text('{"events": [{"job_id": "daily_auto_order", "sta', encoding="utf-8")

    with caplog.at_level("ERROR", logger="scheduler"):
        manager = SchedulerQueueManager(
            _build_once_daily_config(counter), state_file_path=str(state_path)
        )
    assert any(r.levelname == "ERROR" for r in caplog.records)
    quarantined = list(tmp_path.glob("scheduler_queue_state.json.corrupt-*"))
    assert len(quarantined) == 1

    # 2026-10-20 は火曜。当日分も前日分も、記録が無ければ補完の対象になる時刻
    poll_now = datetime(2026, 10, 20, 9, 5, tzinfo=manager.tz)
    assert manager.should_recover("daily_auto_order", now=poll_now) is False
    assert manager.get_missed_past_periods("daily_auto_order", now=poll_now) == []

    # 定時のジョブは止めない。書いた後も補完は止めたまま
    manager.now = lambda: poll_now  # type: ignore[method-assign]
    assert manager.run_job("daily_auto_order", reason="scheduled") is True
    assert counter["n"] == 1
    assert manager.get_missed_past_periods("daily_auto_order", now=poll_now) == []


def test_run_job_error_propagates(tmp_path):
    """func が例外を出した場合、例外が再送出されること"""
    state_path = tmp_path / "state.json"

    def failing_func():
        raise ValueError("意図的なエラー")

    config = {
        "fail_job": {
            "func": failing_func,
            "period": "daily",
            "day_of_week": "*",
            "hour": 8,
            "minute": 0,
            "max_retries": 0,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    with pytest.raises(ValueError, match="意図的なエラー"):
        manager.run_job("fail_job", reason="test")


@pytest.mark.parametrize(
    "exc",
    [SystemExit(1), KeyboardInterrupt()],
    ids=["SystemExit", "KeyboardInterrupt"],
)
@patch("src.orchestration.scheduler_queue.ping_heartbeat")
def test_run_job_base_exception_is_recorded_as_error(mock_ping, tmp_path, exc):
    """Exception でない BaseException で止まった回を success と記録しない（#955）"""
    state_path = tmp_path / "state.json"
    calls = {"n": 0}

    def interrupted_func():
        calls["n"] += 1
        raise exc

    config = {
        "daily_pipeline": {
            "func": interrupted_func,
            "period": "daily",
            "day_of_week": "*",
            "hour": 8,
            "minute": 0,
            "max_retries": 2,
            "retry_base_wait_seconds": 0.0,
            "backfill": True,
        }
    }
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))
    run_now = datetime(2026, 10, 22, 9, 0, tzinfo=manager.tz)
    manager.now = lambda: run_now  # type: ignore[method-assign]

    with pytest.raises(type(exc)):
        manager.run_job("daily_pipeline", reason="test")

    # BaseException はやり直さない
    assert calls["n"] == 1
    with open(state_path, encoding="utf-8") as f:
        events = json.load(f)["events"]
    finished = [e for e in events if e["status"] not in {"started"}]
    assert [e["status"] for e in finished] == ["error"]
    assert finished[0]["error"] == f"{type(exc).__name__}: {exc}"
    assert manager.has_success("daily_pipeline", "2026-10-22") is False
    mock_ping.assert_called_once_with("daily_pipeline", success=False)

    # 前日分の補完の対象に残り、連続失敗にも数える
    next_day = run_now + timedelta(days=1)
    manager.schedule_config["daily_pipeline"]["max_executions_per_period"] = 2
    assert manager.get_missed_past_periods("daily_pipeline", lookback_days=1, now=next_day) == [
        "2026-10-22"
    ]
    from src.utils.alert_service import _count_consecutive_pipeline_failures

    assert _count_consecutive_pipeline_failures(str(state_path)) == 1


# ── Exponential backoff リトライ テスト ──────────────────────────────────────


def _make_failing_config(state_path, max_retries: int, fail_count: int = 999):
    """指定回数だけ失敗し、その後成功する関数を持つ config を返す。"""
    call_count = {"n": 0}

    def _func():
        call_count["n"] += 1
        if call_count["n"] <= fail_count:
            raise RuntimeError(f"transient error #{call_count['n']}")

    return (
        {
            "flaky_job": {
                "func": _func,
                "period": "daily",
                "day_of_week": "*",
                "hour": 8,
                "minute": 0,
                "max_retries": max_retries,
                "retry_base_wait_seconds": 1.0,
            }
        },
        call_count,
    )


@patch("src.orchestration.scheduler_queue.time.sleep")
def test_retry_succeeds_on_second_attempt(mock_sleep, tmp_path):
    """1回失敗後に成功した場合、run_job が True を返し retry イベントが記録されること"""
    state_path = tmp_path / "state.json"
    config, call_count = _make_failing_config(state_path, max_retries=3, fail_count=1)
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    result = manager.run_job("flaky_job", reason="test")

    assert result is True
    assert call_count["n"] == 2

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    statuses = [e["status"] for e in saved["events"]]
    assert "retry" in statuses
    assert statuses[-1] == "success"


@patch("src.orchestration.scheduler_queue.time.sleep")
def test_retry_exhausted_raises_and_sends_notification(mock_sleep, tmp_path):
    """全リトライ失敗後に例外が再送出され、error_notifier が呼ばれること"""
    state_path = tmp_path / "state.json"
    config, call_count = _make_failing_config(state_path, max_retries=2, fail_count=999)
    notifier = MagicMock()
    manager = SchedulerQueueManager(
        config, state_file_path=str(state_path), error_notifier=notifier
    )

    with pytest.raises(RuntimeError):
        manager.run_job("flaky_job", reason="test")

    # func は max_retries+1 回呼ばれる
    assert call_count["n"] == 3

    # notifier が job_id と error_message で呼ばれる
    notifier.assert_called_once()
    args = notifier.call_args[0]
    assert args[0] == "flaky_job"
    assert "transient error" in args[1]

    # retry イベントが 2 件、最終 error イベントが 1 件
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    statuses = [e["status"] for e in saved["events"]]
    assert statuses.count("retry") == 2
    assert statuses[-1] == "error"


@patch("src.orchestration.scheduler_queue.time.sleep")
def test_retry_sleep_uses_exponential_backoff(mock_sleep, tmp_path):
    """sleep が指数的に増加すること（各 wait >= base_wait * 2^attempt）"""
    state_path = tmp_path / "state.json"
    config, _ = _make_failing_config(state_path, max_retries=3, fail_count=999)
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    with pytest.raises(RuntimeError):
        manager.run_job("flaky_job", reason="test")

    assert mock_sleep.call_count == 3
    waits = [call.args[0] for call in mock_sleep.call_args_list]
    base = 1.0
    for i, w in enumerate(waits):
        assert w >= base * (2**i), f"attempt {i}: wait={w} < expected>={base * 2**i}"


@patch("src.orchestration.scheduler_queue.time.sleep")
def test_retry_zero_means_no_retry(mock_sleep, tmp_path):
    """max_retries=0 の場合はリトライなしで即座に例外が再送出されること"""
    state_path = tmp_path / "state.json"
    config, call_count = _make_failing_config(state_path, max_retries=0, fail_count=999)
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    with pytest.raises(RuntimeError):
        manager.run_job("flaky_job", reason="test")

    assert call_count["n"] == 1
    mock_sleep.assert_not_called()


@patch("src.orchestration.scheduler_queue.time.sleep")
def test_non_retryable_error_is_not_retried(mock_sleep, tmp_path):
    """NonRetryableJobError はリトライせず、1回で失敗にして知らせを1回送ること"""
    from src.orchestration.job_errors import NonRetryableJobError

    state_path = tmp_path / "state.json"
    config, call_count = _make_failing_config(state_path, max_retries=3, fail_count=999)
    inner = config["flaky_job"]["func"]

    def _func():
        try:
            inner()
        except RuntimeError as exc:
            raise NonRetryableJobError(str(exc)) from exc

    config["flaky_job"]["func"] = _func
    notifier = MagicMock()
    manager = SchedulerQueueManager(
        config, state_file_path=str(state_path), error_notifier=notifier
    )

    with pytest.raises(NonRetryableJobError):
        manager.run_job("flaky_job", reason="test")

    assert call_count["n"] == 1
    mock_sleep.assert_not_called()
    notifier.assert_called_once()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    statuses = [e["status"] for e in saved["events"]]
    assert "retry" not in statuses
    assert statuses[-1] == "error"


def _schedule_config_with_failing_func(job_id: str):
    """本番の SCHEDULE_CONFIG の job_id の設定のまま、関数だけ毎回落ちるものに替える。"""
    from run_scheduler import SCHEDULE_CONFIG

    call_count = {"n": 0}

    def _func():
        call_count["n"] += 1
        raise RuntimeError(f"failed after placing orders #{call_count['n']}")

    job_config = dict(SCHEDULE_CONFIG[job_id])
    job_config["func"] = _func
    return {job_id: job_config}, call_count


@pytest.mark.parametrize("job_id", ["daily_auto_order", "daily_rule_signals"])
@patch("src.orchestration.scheduler_queue.time.sleep")
def test_order_jobs_in_schedule_config_are_not_retried(mock_sleep, job_id, tmp_path):
    """発注のジョブは同じ回の中で呼び直さず、error を記録して知らせる（二重発注を防ぐ。#836）"""
    state_path = tmp_path / "state.json"
    config, call_count = _schedule_config_with_failing_func(job_id)
    notifier = MagicMock()
    manager = SchedulerQueueManager(
        config, state_file_path=str(state_path), error_notifier=notifier
    )

    with pytest.raises(RuntimeError):
        manager.run_job(job_id, reason="scheduled", force=True)

    assert call_count["n"] == 1
    mock_sleep.assert_not_called()
    notifier.assert_called_once()
    assert notifier.call_args[0][0] == job_id

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    statuses = [e["status"] for e in saved["events"]]
    assert "retry" not in statuses
    assert statuses[-1] == "error"


@patch("src.orchestration.scheduler_queue.time.sleep")
def test_daily_pipeline_in_schedule_config_is_still_retried(mock_sleep, tmp_path):
    """発注でないジョブ（daily_pipeline）は既定の回数までやり直す"""
    from config import settings

    state_path = tmp_path / "state.json"
    config, call_count = _schedule_config_with_failing_func("daily_pipeline")
    notifier = MagicMock()
    manager = SchedulerQueueManager(
        config, state_file_path=str(state_path), error_notifier=notifier
    )

    with pytest.raises(RuntimeError):
        manager.run_job("daily_pipeline", reason="scheduled", force=True)

    assert settings.SCHEDULER_MAX_RETRIES > 0
    assert call_count["n"] == settings.SCHEDULER_MAX_RETRIES + 1
    assert mock_sleep.call_count == settings.SCHEDULER_MAX_RETRIES
    notifier.assert_called_once()


@patch("src.orchestration.scheduler_queue.time.sleep")
def test_retry_notifier_failure_does_not_propagate(mock_sleep, tmp_path):
    """error_notifier 自体が例外を出しても run_job は元の例外を再送出すること"""
    state_path = tmp_path / "state.json"
    config, _ = _make_failing_config(state_path, max_retries=1, fail_count=999)

    def bad_notifier(job_id, error_msg):
        raise ConnectionError("Discord unavailable")

    manager = SchedulerQueueManager(
        config, state_file_path=str(state_path), error_notifier=bad_notifier
    )

    with pytest.raises(RuntimeError, match="transient error"):
        manager.run_job("flaky_job", reason="test")


@patch("src.orchestration.scheduler_queue.time.sleep")
def test_retry_events_recorded_in_state(mock_sleep, tmp_path):
    """retry イベントに attempt / max_retries / wait_seconds / timestamp が含まれること"""
    state_path = tmp_path / "state.json"
    config, _ = _make_failing_config(state_path, max_retries=2, fail_count=999)
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    with pytest.raises(RuntimeError):
        manager.run_job("flaky_job", reason="test")

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    retry_events = [e for e in saved["events"] if e["status"] == "retry"]
    assert len(retry_events) == 2
    for ev in retry_events:
        assert "attempt" in ev
        assert "max_retries" in ev
        assert "wait_seconds" in ev
        assert "timestamp" in ev
        assert ev["job_id"] == "flaky_job"


# ============================================
# 外部死活監視 ping（#496）
# ============================================


def test_run_job_pings_heartbeat_on_success(tmp_path):
    """成功したジョブは job_id を slug に success=True で ping されること"""
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    with patch("src.orchestration.scheduler_queue.ping_heartbeat") as mock_ping:
        assert manager.run_job("daily_pipeline", reason="scheduled") is True

    mock_ping.assert_called_once_with("daily_pipeline", success=True)


def test_run_job_pings_heartbeat_fail_on_error(tmp_path):
    """失敗したジョブは success=False で ping されること（/fail 通知）"""

    def _failing_func():
        raise RuntimeError("intentional failure")

    config = {
        "daily_pipeline": {
            "func": _failing_func,
            "period": "daily",
            "day_of_week": "mon-fri",
            "hour": 19,
            "minute": 0,
            "recovery_delay_minutes": 10,
            "max_executions_per_period": 1,
            "max_retries": 0,
        }
    }
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(config, state_file_path=str(state_path))

    with patch("src.orchestration.scheduler_queue.ping_heartbeat") as mock_ping:
        with pytest.raises(RuntimeError, match="intentional failure"):
            manager.run_job("daily_pipeline", reason="manual", force=True)

    mock_ping.assert_called_once_with("daily_pipeline", success=False)


def test_run_job_does_not_ping_on_skip(tmp_path):
    """実行上限スキップ時は ping しないこと（skip を生存信号にしない）"""
    counter = {}
    state_path = tmp_path / "scheduler_queue_state.json"
    manager = SchedulerQueueManager(_build_config(counter), state_file_path=str(state_path))

    manager.run_job("daily_pipeline", reason="scheduled")
    manager.run_job("daily_pipeline", reason="recovery")

    with patch("src.orchestration.scheduler_queue.ping_heartbeat") as mock_ping:
        assert manager.run_job("daily_pipeline", reason="recovery") is False

    mock_ping.assert_not_called()
