"""results/scheduler_queue_state.json のジョブの実行の記録を回ごとの表にするテスト（#1892）。"""

import json
import logging

import pandas as pd

from src.reporting.scheduler_run_trend import (
    COLUMNS,
    build_scheduler_run_trend,
    filter_scheduler_runs,
    format_scheduler_run_csv,
    format_scheduler_run_markdown,
    load_scheduler_events,
    load_scheduler_run_trend,
)


def _started(job, period, run_id, at):
    return {
        "job_id": job,
        "reason": "cron",
        "status": "started",
        "period_key": period,
        "run_id": run_id,
        "boot_id": "b1",
        "started_at": at,
    }


def _finished(job, period, run_id, status, start, end, duration, error=None):
    event = {
        "job_id": job,
        "reason": "cron",
        "status": status,
        "period_key": period,
        "started_at": start,
        "finished_at": end,
        "duration_seconds": duration,
        "error": error,
    }
    if run_id is not None:
        event["run_id"] = run_id
    return event


def _retry(job, period, attempt):
    return {
        "job_id": job,
        "reason": "cron",
        "status": "retry",
        "period_key": period,
        "attempt": attempt,
        "max_retries": 3,
        "error": "boom",
        "wait_seconds": 1.0,
        "timestamp": "2026-11-01T00:00:00+00:00",
    }


def test_started_and_success_pair_is_one_row():
    df = build_scheduler_run_trend(
        [
            _started("daily_pipeline", "2026-11-10", "r1", "2026-11-10T07:00:00+00:00"),
            _finished(
                "daily_pipeline",
                "2026-11-10",
                "r1",
                "success",
                "2026-11-10T07:00:00+00:00",
                "2026-11-10T07:10:00+00:00",
                600.1234,
            ),
        ]
    )
    assert list(df.columns) == COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert row["run_id"] == "r1"
    assert row["status"] == "success"
    assert row["finished_at"] == "2026-11-10T07:10:00+00:00"
    assert row["duration_seconds"] == 600.123
    assert row["retries"] == 0
    assert not row["missing_finish"]
    assert pd.isna(row["error"])


def test_retries_are_counted_on_the_open_run():
    job, period = "weekly_rule_evaluation", "2026-W44"
    df = build_scheduler_run_trend(
        [
            _started(job, period, "r1", "2026-11-01T00:00:00+00:00"),
            _retry(job, period, 1),
            _retry(job, period, 2),
            _retry(job, period, 3),
            _finished(
                job,
                period,
                "r1",
                "error",
                "2026-11-01T00:00:00+00:00",
                "2026-11-01T00:09:13+00:00",
                553.0,
                error="Traceback line\nsecond line",
            ),
        ]
    )
    assert len(df) == 1
    row = df.iloc[0]
    assert row["retries"] == 3
    assert row["status"] == "error"
    assert row["error"] == "Traceback line"


def test_started_without_finish_is_marked():
    job, period = "weekly_model_training", "2026-W46"
    df = build_scheduler_run_trend(
        [
            _started(job, period, "gone", "2026-11-10T01:00:00+00:00"),
            _retry(job, period, 1),
            _started(job, period, "r2", "2026-11-10T03:00:00+00:00"),
            _finished(
                job,
                period,
                "r2",
                "success",
                "2026-11-10T03:00:00+00:00",
                "2026-11-10T04:00:00+00:00",
                3600.0,
            ),
        ]
    )
    assert len(df) == 2
    gone = df[df["run_id"] == "gone"].iloc[0]
    assert gone["missing_finish"]
    assert pd.isna(gone["status"])
    assert pd.isna(gone["finished_at"])
    assert gone["retries"] == 1
    done = df[df["run_id"] == "r2"].iloc[0]
    assert not done["missing_finish"]
    assert done["retries"] == 0

    unfinished = filter_scheduler_runs(df, status="unfinished")
    assert list(unfinished["run_id"]) == ["gone"]


def test_old_records_without_run_id_are_one_row_each():
    job = "strategy_promotion_check"
    df = build_scheduler_run_trend(
        [
            _finished(
                job,
                "2026-03-05",
                None,
                "success",
                "2026-03-05T13:18:12+00:00",
                "2026-03-05T13:18:12+00:00",
                0.0,
            ),
            _retry(job, "2026-03-06", 1),
            _finished(
                job,
                "2026-03-06",
                None,
                "error",
                "2026-03-06T13:18:12+00:00",
                "2026-03-06T13:19:12+00:00",
                60.0,
                error="x",
            ),
            _finished(
                job,
                "2026-03-06",
                None,
                "skipped_limit",
                "2026-03-06T14:00:00+00:00",
                "2026-03-06T14:00:00+00:00",
                0.0,
            ),
        ]
    )
    assert len(df) == 3
    assert list(df["status"]) == ["success", "error", "skipped_limit"]
    assert list(df["retries"]) == [0, 1, 0]
    assert df["run_id"].isna().all()
    assert not df["missing_finish"].any()


def test_finish_without_started_is_one_row():
    df = build_scheduler_run_trend(
        [
            _finished(
                "daily_pipeline",
                "2026-11-01",
                "cut",
                "success",
                "2026-11-01T07:00:00+00:00",
                "2026-11-01T07:01:00+00:00",
                60.0,
            )
        ]
    )
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "cut"
    assert df.iloc[0]["started_at"] == "2026-11-01T07:00:00+00:00"


def test_broken_json_warns_and_returns_empty_table(tmp_path, caplog):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    wrong = tmp_path / "wrong.json"
    wrong.write_text("[1, 2]", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        df = load_scheduler_run_trend(str(broken))
        assert load_scheduler_events(str(wrong)) == []
        assert load_scheduler_events(str(tmp_path / "missing.json")) == []
    assert df.empty
    assert list(df.columns) == COLUMNS
    assert sum(r.levelno == logging.WARNING for r in caplog.records) >= 3
    assert format_scheduler_run_markdown(df) == "データなし"


def test_load_from_file_and_filters_and_formats(tmp_path):
    events = [
        _started("daily_pipeline", "2026-11-09", "a", "2026-11-09T07:00:00+00:00"),
        _finished(
            "daily_pipeline",
            "2026-11-09",
            "a",
            "error",
            "2026-11-09T07:00:00+00:00",
            "2026-11-09T07:00:05+00:00",
            5.0,
            error="a|b",
        ),
        _started("weekly_model_training", "2026-W46", "b", "2026-11-11T07:00:00+00:00"),
        _finished(
            "weekly_model_training",
            "2026-W46",
            "b",
            "success",
            "2026-11-11T07:00:00+00:00",
            "2026-11-11T08:00:00+00:00",
            3600.0,
        ),
        "not a dict",
    ]
    path = tmp_path / "scheduler_queue_state.json"
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    df = load_scheduler_run_trend(str(path))
    assert len(df) == 2

    assert list(filter_scheduler_runs(df, job="daily_pipeline")["run_id"]) == ["a"]
    assert list(filter_scheduler_runs(df, since="2026-11-10")["run_id"]) == ["b"]
    assert list(filter_scheduler_runs(df, status="error")["run_id"]) == ["a"]
    assert len(filter_scheduler_runs(df)) == 2

    markdown = format_scheduler_run_markdown(df)
    assert markdown.splitlines()[0].startswith("| job_id | period_key | run_id")
    assert "a\\|b" in markdown
    assert "null" in markdown
    csv = format_scheduler_run_csv(df)
    assert csv.splitlines()[0] == ",".join(COLUMNS)
