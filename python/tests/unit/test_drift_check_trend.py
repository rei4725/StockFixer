"""results/drift_check の記録を日ごとに1行の表にする読み手のテスト（#2050）。"""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from src.orchestration.jobs.drift_record import DriftCheckRecord, write_drift_check_record
from src.reporting.drift_check_trend import TREND_COLUMNS, load_drift_check_trend


def _write(tmp_path, day: int, n_triggered: int, retrain: int) -> None:
    record = DriftCheckRecord(
        n_evaluated=781,
        n_triggered=n_triggered,
        triggered_share=n_triggered / 781,
        warn_share=0.15,
        warn=False,
        mae_threshold=0.03,
        hit_rate_threshold=0.4,
        fresh_skipped=3,
        retrain_targets=retrain,
        retrain_success=retrain,
    )
    now = datetime(2026, 11, day, 23, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    write_drift_check_record(record, out_dir=str(tmp_path), now=now)


def test_one_row_per_day_in_date_order(tmp_path):
    """複数の日の JSON から、日の順に1日1行の表ができる。"""
    _write(tmp_path, 18, 34, 11)
    _write(tmp_path, 12, 16, 5)
    _write(tmp_path, 17, 32, 17)

    df = load_drift_check_trend(str(tmp_path))

    assert list(df.columns) == TREND_COLUMNS
    assert list(df["date"]) == ["2026-11-12", "2026-11-17", "2026-11-18"]
    assert list(df["n_triggered"]) == [16, 32, 34]
    assert list(df["retrain_success"]) == [5, 17, 11]
    assert list(df["n_evaluated"]) == [781, 781, 781]


def test_broken_and_foreign_files_are_skipped(tmp_path):
    """壊れた JSON・オブジェクトでない JSON・名前の形が違うファイルは飛ばす。"""
    _write(tmp_path, 15, 19, 4)
    (tmp_path / "2026-11-16.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "2026-11-17.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    (tmp_path / "notes.json").write_text(json.dumps({"n_triggered": 1}), encoding="utf-8")

    df = load_drift_check_trend(str(tmp_path))

    assert list(df["date"]) == ["2026-11-15"]
    assert df.loc[0, "n_triggered"] == 19


def test_missing_keys_are_empty_and_empty_dir_gives_empty_table(tmp_path):
    """無い項目は空になり、記録の無い置き場からは空の表が返る。"""
    assert load_drift_check_trend(str(tmp_path)).empty
    (tmp_path / "2026-11-19.json").write_text(json.dumps({"n_evaluated": 700}), encoding="utf-8")

    df = load_drift_check_trend(str(tmp_path))

    assert df.loc[0, "n_evaluated"] == 700
    assert df.loc[0, "retrain_success"] is None or df["retrain_success"].isna().all()
