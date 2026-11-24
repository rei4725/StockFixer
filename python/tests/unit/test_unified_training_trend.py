"""results/unified_training の記録を1回の学習ごとに1行の表にする読み手のテスト（#2151）。"""

import json
import sys
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src.prediction import training_record
from src.prediction.training_record import UnifiedTrainingRecord, write_training_record
from src.reporting import unified_training_trend as trend
from src.reporting.unified_training_trend import (
    TREND_COLUMNS,
    format_unified_training_csv,
    format_unified_training_markdown,
    load_unified_training_trend,
)

XGB = "UnifiedStockXGBoost_challenger"
LGB = "UnifiedStockLightGBM_challenger"


def _write(tmp_path, date: str, model: str, **overrides) -> None:
    payload = {
        "model_name": model,
        "best_iteration": 12,
        "trees": 500,
        "val_rows": 173433,
        "val_pred_std": 0.005,
        "is_constant": False,
        "recorded_at": f"{date}T10:00:00+00:00",
        "val_predicted_up_rate": 0.25,
        "val_actual_up_rate": 0.5,
        **overrides,
    }
    (tmp_path / f"{date}_{model}.json").write_text(json.dumps(payload), encoding="utf-8")


def _cells(df, row: int) -> dict:
    """Markdown の row 行目を、列の名前と文字の組にする。"""
    line = format_unified_training_markdown(df).splitlines()[2 + row]
    return dict(zip(TREND_COLUMNS, line.strip("| ").split(" | ")))


def test_rows_are_ordered_by_date_and_model_with_gap(tmp_path):
    """2日・2モデルの記録が日付・モデル名の順に並び、ずれは予想 − 実際になる。"""
    _write(tmp_path, "2026-11-25", XGB, val_predicted_up_rate=0.5)
    _write(tmp_path, "2026-11-25", LGB, val_predicted_up_rate=0.75)
    _write(tmp_path, "2026-11-24", XGB, val_predicted_up_rate=0.25)
    _write(tmp_path, "2026-11-24", LGB, val_predicted_up_rate=0.625)

    df = load_unified_training_trend(str(tmp_path))

    assert list(df.columns) == TREND_COLUMNS
    assert list(zip(df["date"], df["model_name"])) == [
        ("2026-11-24", LGB),
        ("2026-11-24", XGB),
        ("2026-11-25", LGB),
        ("2026-11-25", XGB),
    ]
    assert list(df["up_rate_gap"]) == [0.125, -0.25, 0.25, 0.0]
    assert df.loc[0, "recorded_at"] == "2026-11-24T10:00:00+00:00"


def test_run_without_validation_rows_shows_null_not_zero_or_false(tmp_path):
    """検証の行が無い回は null と出て、検証の行がある回の False・0 と分かれる。"""
    _write(
        tmp_path,
        "2026-11-24",
        XGB,
        val_rows=0,
        val_pred_std=None,
        is_constant=None,
        val_predicted_up_rate=None,
        val_actual_up_rate=None,
    )
    _write(tmp_path, "2026-11-25", XGB, val_predicted_up_rate=0.0, val_actual_up_rate=0.0)
    df = load_unified_training_trend(str(tmp_path))

    missing, present = _cells(df, 0), _cells(df, 1)
    for column in TREND_COLUMNS[6:]:
        assert missing[column] == "null", column
    assert missing["val_rows"] == "0"
    assert present["val_rows"] == "173433"
    assert present["is_constant"] == "False"
    assert present["val_predicted_up_rate"] == "0"
    assert present["up_rate_gap"] == "0"

    lines = format_unified_training_csv(df).splitlines()
    assert lines[1].endswith(",12,500,0,null,null,null,null,null")
    assert lines[2].endswith(",12,500,173433,0.005,False,0.0,0.0,0.0")


@pytest.mark.parametrize("missing", ["val_predicted_up_rate", "val_actual_up_rate"])
def test_gap_is_null_when_either_rate_is_missing(tmp_path, missing):
    """どちらか一方の割合だけが無くても、ずれは null になる。"""
    _write(tmp_path, "2026-11-24", XGB, **{missing: None})

    assert load_unified_training_trend(str(tmp_path)).loc[0, "up_rate_gap"] is None


def test_best_iteration_zero_and_null_are_distinguished(tmp_path):
    """best_iteration の 0 と null が、Markdown でも CSV でも分かれて出る。"""
    _write(tmp_path, "2026-11-24", LGB, best_iteration=0)
    _write(tmp_path, "2026-11-24", XGB, best_iteration=None)
    df = load_unified_training_trend(str(tmp_path))

    assert _cells(df, 0)["best_iteration"] == "0"
    assert _cells(df, 1)["best_iteration"] == "null"
    lines = format_unified_training_csv(df).splitlines()
    assert lines[1].split(",")[3] == "0"
    assert lines[2].split(",")[3] == "null"


def test_broken_and_foreign_files_are_skipped_with_warning(tmp_path):
    """名前の形が違う・壊れた・オブジェクトでない・読めないものは、warning を出して飛ばす。"""
    _write(tmp_path, "2026-11-24", XGB)
    bad = {
        "notes.json": b"{}",
        "2026-11-24.json": b"{}",
        "2026-11-25_Broken.json": b"{not json",
        "2026-11-25_Bytes.json": b"\xff\xfe",
        "2026-11-25_List.json": b"[1, 2]",
    }
    for name, content in bad.items():
        (tmp_path / name).write_bytes(content)

    with patch.object(trend.logger, "warning") as warning:
        df = load_unified_training_trend(str(tmp_path))

    assert list(df["date"]) == ["2026-11-24"]
    warned = [str(call.args[1]) for call in warning.call_args_list]
    assert sorted(warned) == sorted(str(tmp_path / name) for name in bad)


def test_empty_or_missing_dir_gives_no_data(tmp_path):
    """置き場が空でも無くても、表は「データなし」になる。"""
    for record_dir in (tmp_path, tmp_path / "missing"):
        df = load_unified_training_trend(str(record_dir))
        assert list(df.columns) == TREND_COLUMNS
        assert format_unified_training_markdown(df) == "データなし"


def test_reads_what_the_writer_writes(tmp_path, monkeypatch):
    """write_training_record が書いた JSON を、そのまま1行に読む。"""
    now = datetime(2026, 11, 24, 8, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    monkeypatch.setattr(training_record, "get_training_record_dir", lambda: str(tmp_path))
    monkeypatch.setattr(training_record, "now_jst", lambda: now)
    record = UnifiedTrainingRecord(
        model_name=XGB,
        best_iteration=None,
        trees=500,
        val_rows=4,
        val_pred_std=0.00547,
        is_constant=False,
    )
    write_training_record(record, val_pred=[0.1, -0.1, -0.1, -0.1], val_actual=[0.1, 0.1, -0.1, 0])

    df = load_unified_training_trend(str(tmp_path))

    assert len(df) == 1
    row = df.loc[0].to_dict()
    recorded_at = row.pop("recorded_at")
    assert datetime.fromisoformat(recorded_at.replace("Z", "+00:00")) == now
    assert row == {
        "date": "2026-11-24",
        "model_name": XGB,
        "best_iteration": None,
        "trees": 500,
        "val_rows": 4,
        "val_pred_std": 0.00547,
        "is_constant": False,
        "val_predicted_up_rate": 0.25,
        "val_actual_up_rate": 0.5,
        "up_rate_gap": -0.25,
    }


def test_cli_model_and_tail(tmp_path, monkeypatch, capsys):
    """CLI は --model で絞り、--tail を絞った後の行に当てる。"""
    import run_unified_training_trend as cli

    records = tmp_path / "records"
    records.mkdir()
    for date in ("2026-11-23", "2026-11-24", "2026-11-25"):
        _write(records, date, LGB)
        _write(records, date, XGB)
    base = ["run_unified_training_trend.py", "--dir", str(records)]

    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    assert len(capsys.readouterr().out.splitlines()) == 8

    monkeypatch.setattr(sys, "argv", base + ["--model", LGB])
    cli.main()
    text = capsys.readouterr().out
    assert len(text.splitlines()) == 5 and XGB not in text

    out = tmp_path / "out.csv"
    argv = base + ["--model", LGB, "--tail", "2", "--format", "csv", "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(TREND_COLUMNS)
    assert [line.split(",")[:2] for line in lines[1:]] == [["2026-11-24", LGB], ["2026-11-25", LGB]]

    monkeypatch.setattr(sys, "argv", base + ["--model", "NoSuchModel"])
    cli.main()
    assert capsys.readouterr().out == "データなし\n"


def test_cli_reads_unified_training_under_results_by_default(tmp_path, monkeypatch, capsys):
    """--dir を省くと、results の下の unified_training を読む。"""
    import run_unified_training_trend as cli

    records = tmp_path / training_record.TRAINING_RECORD_SUBDIR
    records.mkdir()
    _write(records, "2026-11-24", XGB)
    monkeypatch.setattr(cli, "get_results_dir", lambda: str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["run_unified_training_trend.py"])
    cli.main()

    assert f"| 2026-11-24 | {XGB} |" in capsys.readouterr().out
