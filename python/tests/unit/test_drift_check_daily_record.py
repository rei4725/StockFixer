"""日次ドリフトチェックがその日の判定と再学習の数を results/drift_check に残すことのテスト（#2050）。"""

import json
import os
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from src.domain.types import SymbolTask
from src.orchestration.jobs.drift_check import run_daily_drift_check
from src.orchestration.jobs.drift_record import (
    DriftCheckRecord,
    get_drift_record_dir,
    write_drift_check_record,
)
from src.utils.data_path_utils import get_results_dir


def _row(symbol: str, mae: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "mean_abs_error": mae,
        "direction_accuracy": 0.55,
        "n_samples": 20,
    }


def _read_records() -> list[dict]:
    record_dir = get_drift_record_dir()
    if not os.path.isdir(record_dir):
        return []
    records = []
    for name in sorted(os.listdir(record_dir)):
        with open(os.path.join(record_dir, name), encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def test_record_dir_is_under_isolated_results():
    """記録の置き場は get_results_dir() の下（unit テストでは conftest の一時フォルダ）。"""
    assert get_drift_record_dir() == os.path.join(get_results_dir(), "drift_check")
    assert os.environ["STOCKFIXER_RESULTS_DIR"] == get_results_dir()


def test_write_record_uses_jst_date_and_overwrites_same_day(tmp_path):
    """JST の日付の名前で書き、同じ日のやり直しは上書きする。"""
    record = DriftCheckRecord(
        n_evaluated=781,
        n_triggered=34,
        triggered_share=34 / 781,
        warn_share=0.15,
        warn=False,
        mae_threshold=0.03,
        hit_rate_threshold=0.4,
    )
    now = datetime(2026, 11, 18, 23, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    path = write_drift_check_record(record, out_dir=str(tmp_path), now=now)
    assert path == os.path.join(str(tmp_path), "2026-11-18.json")

    record.retrain_targets = 11
    record.retrain_success = 11
    write_drift_check_record(record, out_dir=str(tmp_path), now=now)
    assert os.listdir(tmp_path) == ["2026-11-18.json"]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["date"] == "2026-11-18"
    assert data["n_triggered"] == 34
    assert data["retrain_success"] == 11


def test_write_record_failure_returns_none(tmp_path):
    """書けないときは例外を外に出さず None を返す。"""
    blocker = tmp_path / "file"
    blocker.write_text("x")
    record = DriftCheckRecord(1, 0, 0.0, 0.15, False, 0.02, 0.45)
    assert write_drift_check_record(record, out_dir=str(blocker / "sub")) is None


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_job_records_counts_and_retrain(
    mock_load_summary, mock_notify, mock_load_symbols, mock_train, mock_streak, tmp_path
):
    """判定の数と再学習の対象・成功の数が JSON に入る。"""
    rows = [_row(f"S{i}", 0.058) for i in range(8)] + [_row("BAD1", 0.16), _row("BAD2", 0.15)]
    mock_load_summary.return_value = pd.DataFrame(rows)
    mock_load_symbols.return_value = [SymbolTask(market="jp", symbol=r["symbol"]) for r in rows]
    mock_train.side_effect = [{"status": "success"}, {"status": "skipped"}]

    with patch("src.utils.data_path_utils.get_models_dir", return_value=str(tmp_path)):
        run_daily_drift_check()

    records = _read_records()
    assert len(records) == 1
    record = records[0]
    assert record["n_evaluated"] == 10
    assert record["n_triggered"] == 2
    assert record["triggered_share"] == 0.2
    warn, share, n_triggered, n_evaluated = mock_streak.call_args.args
    assert (record["warn"], record["n_triggered"], record["n_evaluated"]) == (
        warn,
        n_triggered,
        n_evaluated,
    )
    assert record["triggered_share"] == share
    assert record["fresh_skipped"] == 0
    assert record["retrain_targets"] == 2
    assert record["retrain_success"] == 1
    assert record["mae_threshold"] > 0
    assert record["hit_rate_threshold"] > 0


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_job_records_day_without_triggered(
    mock_load_summary, mock_notify, mock_train, mock_streak, tmp_path
):
    """閾値超過が0で早く戻る日も記録を書く（再学習の数は0）。"""
    mock_load_summary.return_value = pd.DataFrame([_row(f"S{i}", 0.01) for i in range(10)])

    with patch("src.utils.data_path_utils.get_models_dir", return_value=str(tmp_path)):
        run_daily_drift_check()

    mock_train.assert_not_called()
    mock_notify.assert_not_called()
    records = _read_records()
    assert len(records) == 1
    assert records[0]["n_evaluated"] == 10
    assert records[0]["n_triggered"] == 0
    assert records[0]["retrain_targets"] == 0
    assert records[0]["retrain_success"] == 0


@patch("src.orchestration.jobs.drift_record.json.dump", side_effect=OSError("disk full"))
@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_job_continues_to_retrain_when_record_write_fails(
    mock_load_summary,
    mock_notify,
    mock_load_symbols,
    mock_train,
    mock_streak,
    mock_dump,
    tmp_path,
    caplog,
):
    """記録を書けなくても再学習まで進み、logger.error に残す。"""
    rows = [_row(f"S{i}", 0.058) for i in range(8)] + [_row("BAD", 0.16)]
    mock_load_summary.return_value = pd.DataFrame(rows)
    mock_load_symbols.return_value = [SymbolTask(market="jp", symbol=r["symbol"]) for r in rows]
    mock_train.return_value = {"status": "success"}

    with patch("src.utils.data_path_utils.get_models_dir", return_value=str(tmp_path)):
        run_daily_drift_check()

    assert [c.args[0].symbol for c in mock_train.call_args_list] == ["BAD"]
    mock_notify.assert_called_once()
    mock_dump.assert_called()
    assert any(
        "日次の記録を書けず" in r.getMessage() and r.levelname == "ERROR" for r in caplog.records
    )
