"""日次ドリフトチェックの記録に超えた銘柄の一覧と再学習の所要秒が残ることのテスト（#2167）。"""

import json
import os
from collections import Counter
from unittest.mock import patch

import pandas as pd
import pytest

from src.domain.types import SymbolTask
from src.orchestration.jobs.drift_check import run_daily_drift_check
from src.orchestration.jobs.drift_record import get_drift_record_dir

_JOB = "src.orchestration.jobs.drift_check"
_RECORD = "src.orchestration.jobs.drift_record"


def _row(market: str, symbol: str, mae: float, hit_rate: float = 0.55) -> dict:
    return {
        "market": market,
        "symbol": symbol,
        "mean_abs_error": mae,
        "direction_accuracy": hit_rate,
        "n_samples": 20,
    }


# 中央値は MAE 0.02・Hit Rate 0.55 のまま。線は MAE 0.03、Hit Rate 0.35
_HEALTHY = [_row("jp", f"S{i}", 0.02) for i in range(20)]
_BAD = [
    _row("us", "HITONLY", 0.021, 0.30),
    _row("jp", "OK", 0.19),
    _row("jp", "FRESH", 0.20),
    _row("us", "SKIPPED", 0.18),
    _row("us", "RAISED", 0.17),
    _row("us", "NOTASK", 0.16),
    _row("us", "BOTH", 0.15, 0.20),
]


def _read_record() -> dict:
    record_dir = get_drift_record_dir()
    names = os.listdir(record_dir)
    assert len(names) == 1
    with open(os.path.join(record_dir, names[0]), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def job(monkeypatch):
    """外の世界を差し替えたジョブ。rows を渡して流し、差し替えた口を返す。"""
    monkeypatch.setenv("DRIFT_MAX_RETRAIN_PER_RUN", "5")

    def _train(task):
        if task.symbol == "RAISED":
            raise RuntimeError("train failed")
        return {"status": "skipped" if task.symbol == "SKIPPED" else "success"}

    with (
        patch(f"{_JOB}.record_drift_warn_streak") as streak,
        patch(f"{_JOB}._is_recently_retrained", side_effect=lambda m, s: s == "FRESH"),
        patch("src.prediction.training_pipeline.train_models_for_symbol_task") as train,
        patch("src.watchlist.batch_runner.load_target_symbols") as load_symbols,
        patch("src.reporting.discord.discord_utils.send_drift_retrain_notification") as notify,
        patch("src.reporting.discord.discord_utils.send_feature_suggestion_notification"),
        patch("src.prediction.db.load_feature_exclusion_candidates", return_value=[]),
        patch("src.prediction.db.load_drift_summary") as load_summary,
    ):
        train.side_effect = _train

        def _run(rows):
            load_summary.return_value = pd.DataFrame(rows)
            load_symbols.return_value = [
                SymbolTask(market=r["market"], symbol=r["symbol"])
                for r in rows
                if r["symbol"] != "NOTASK"
            ]
            run_daily_drift_check()
            return {"train": train, "notify": notify, "streak": streak}

        yield _run


def _assert_job_unchanged(mocks) -> None:
    """再学習に渡す銘柄と順・上限・知らせ・連続回数に渡す値が決めたとおり。"""
    trained = [c.args[0].symbol for c in mocks["train"].call_args_list]
    assert trained == ["OK", "SKIPPED", "RAISED", "BOTH"]
    mocks["notify"].assert_called_once()
    symbols, mae_threshold, hit_rate_threshold = mocks["notify"].call_args.args
    assert [s["symbol"] for s in symbols] == ["FRESH", "OK", "SKIPPED", "RAISED", "NOTASK"]
    assert all(
        set(s) == {"market", "symbol", "mean_abs_error", "direction_accuracy"} for s in symbols
    )
    assert mae_threshold == pytest.approx(0.03)
    assert hit_rate_threshold == pytest.approx(0.35)
    mocks["streak"].assert_called_once_with(True, 7 / 27, 7, 27)


def test_job_values_are_unchanged(job):
    """一覧を足しても、再学習・知らせ・連続回数に渡す値は変わらない。"""
    _assert_job_unchanged(job(_HEALTHY + _BAD))


def test_record_lists_triggered_symbols_with_disposition(job):
    """成功・失敗・例外・タスクなし・上限の外・直近の学び直しが、数の鍵と合う。"""
    job(_HEALTHY + _BAD)
    record = _read_record()

    listed = record["triggered_symbols"]
    assert len(listed) == record["n_triggered"] == 7
    assert {s["symbol"]: s["disposition"] for s in listed} == {
        "FRESH": "fresh_skipped",
        "OK": "retrained",
        "SKIPPED": "retrain_failed",
        "RAISED": "retrain_failed",
        "NOTASK": "retrain_failed",
        "BOTH": "retrained",
        "HITONLY": "over_cap",
    }
    counts = Counter(s["disposition"] for s in listed)
    assert counts["fresh_skipped"] == record["fresh_skipped"] == 1
    assert counts["retrained"] == record["retrain_success"] == 2
    assert counts["retrained"] + counts["retrain_failed"] == record["retrain_targets"] == 5
    by_symbol = {s["symbol"]: s for s in listed}
    assert by_symbol["BOTH"] == {
        "market": "us",
        "symbol": "BOTH",
        "mean_abs_error": 0.15,
        "direction_accuracy": 0.20,
        "mae_over": True,
        "hit_rate_over": True,
        "disposition": "retrained",
    }
    assert (by_symbol["OK"]["mae_over"], by_symbol["OK"]["hit_rate_over"]) == (True, False)
    hit_only = by_symbol["HITONLY"]
    assert (hit_only["mae_over"], hit_only["hit_rate_over"]) == (False, True)


def test_flags_include_equality_like_selection(job):
    """線とちょうど等しい銘柄も、選び出しの判定と同じく超えたと記す。"""
    job(_HEALTHY + [_row("jp", "MAE", 0.02 * 1.5), _row("us", "HIT", 0.01, 0.55 - 0.20)])

    flags = {
        s["symbol"]: (s["mae_over"], s["hit_rate_over"])
        for s in _read_record()["triggered_symbols"]
    }
    assert flags == {"MAE": (True, False), "HIT": (False, True)}


def test_nan_is_written_as_null(job):
    """値の欠けた側は超えておらず、JSON には null で書く。"""
    job(_HEALTHY + [_row("us", "NAN", float("nan"), 0.20)])

    (entry,) = _read_record()["triggered_symbols"]
    assert entry["mean_abs_error"] is None
    assert (entry["mae_over"], entry["hit_rate_over"]) == (False, True)


def test_retrain_seconds_is_elapsed_of_retrain_loop(job):
    """再学習の並びの前後の差を秒で残す。"""
    with patch(f"{_RECORD}.time") as mock_time:
        mock_time.monotonic.side_effect = [100.0, 2260.5]
        job(_HEALTHY + _BAD)
    assert _read_record()["retrain_seconds"] == 2160.5


def test_no_retrain_target_day_has_null_seconds(job):
    """超過があっても再学習の対象が無い日の秒は null。"""
    mocks = job(_HEALTHY + [_row("jp", "FRESH", 0.20)])
    mocks["train"].assert_not_called()
    record = _read_record()
    assert record["retrain_seconds"] is None
    assert [s["disposition"] for s in record["triggered_symbols"]] == ["fresh_skipped"]


def test_day_without_triggered_has_empty_list_and_null_seconds(job):
    """超過0で早く戻る日は、空の一覧と null の秒を書く。"""
    job(_HEALTHY)
    record = _read_record()
    assert record["n_triggered"] == 0
    assert record["triggered_symbols"] == []
    assert record["retrain_seconds"] is None


def test_detail_failure_keeps_retrain_and_counts(job, caplog):
    """一覧を組む所が例外を出しても、再学習と今までの数の記録は残る。"""
    with patch(f"{_RECORD}.DriftTriggeredSymbol", side_effect=RuntimeError("broken")):
        mocks = job(_HEALTHY + _BAD)
    _assert_job_unchanged(mocks)
    record = _read_record()
    assert record["triggered_symbols"] is None
    assert (record["n_triggered"], record["retrain_targets"], record["retrain_success"]) == (
        7,
        5,
        2,
    )
    assert any(r.levelname == "ERROR" and "一覧" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("readings", [[RuntimeError("clock")], [1.0, RuntimeError("clock")]])
def test_timer_failure_keeps_retrain_and_record(job, readings):
    """時計が例外を出しても、再学習と記録は進み、秒だけが null になる。"""
    with patch(f"{_RECORD}.time") as mock_time:
        mock_time.monotonic.side_effect = readings
        mocks = job(_HEALTHY + _BAD)
    _assert_job_unchanged(mocks)
    record = _read_record()
    assert record["retrain_seconds"] is None
    assert len(record["triggered_symbols"]) == 7
