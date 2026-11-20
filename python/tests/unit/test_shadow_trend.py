"""results/accuracy/shadow の週ごとの JSON を読む表のテスト（#1863）。"""

import json
import math

import pytest

from src.reporting.shadow_trend import (
    CONSTANT_STREAK,
    DIFF_COLUMN,
    NOT_WIN_STREAK,
    SHADOW_COLUMNS,
    SHADOW_ERROR_HEAD,
    build_shadow_trend,
    format_shadow_markdown,
    load_shadow_records,
    load_shadow_trend,
)


def _full(week, prod, chal, wins, constant, skipped=None):
    return {
        "week": week,
        "written_at": f"{week}T00:00:00+00:00",
        "window_days": 4,
        "production_hit_rate": prod,
        "challenger_hit_rate": chal,
        "production_sharpe": 0.5,
        "challenger_sharpe": 0.1,
        "n_production": 100,
        "n_challenger": 90,
        "challenger_wins": wins,
        "challenger_prediction_std": 1.18e-16 if constant else 0.02,
        "challenger_constant": constant,
        "gate_skipped_reason": skipped,
        "promoted": False,
        "retry_count": 1,
    }


def _write(dir_path, name, payload):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (dir_path / f"{name}.json").write_text(text, encoding="utf-8")


@pytest.fixture
def shadow_dir(tmp_path):
    # 古い週（欄の欠けた記録）、勝った週、定数で飛ばした週が2つ、壊れたファイル、形の違う JSON
    _write(tmp_path, "2026-W42", {"week": "2026-W42", "promoted": False})
    _write(tmp_path, "2026-W43", _full("2026-W43", 0.40, 0.45, True, False))
    _write(tmp_path, "2026-W45", _full("2026-W45", 0.42, 0.38, False, True, "challenger_constant"))
    _write(tmp_path, "2026-W44", _full("2026-W44", 0.41, 0.39, False, False))
    _write(tmp_path, "2026-W46", _full("2026-W46", 0.396, 0.30, False, True, "challenger_constant"))
    _write(tmp_path, "2026-W47", "{not json")
    _write(tmp_path, "2026-W48", "[1, 2]")
    return tmp_path


def test_loads_only_readable_records(shadow_dir):
    records = load_shadow_records(str(shadow_dir))
    assert [r["week"] for r in records] == [
        "2026-W42",
        "2026-W43",
        "2026-W44",
        "2026-W45",
        "2026-W46",
    ]


def test_week_falls_back_to_file_name(tmp_path):
    _write(tmp_path, "2026-W40", {"challenger_wins": True})
    assert load_shadow_records(str(tmp_path))[0]["week"] == "2026-W40"


def test_table_has_one_row_per_week_in_order(shadow_dir):
    df = load_shadow_trend(str(shadow_dir))
    assert list(df.columns) == SHADOW_COLUMNS
    assert list(df["week"]) == ["2026-W42", "2026-W43", "2026-W44", "2026-W45", "2026-W46"]
    w46 = df[df["week"] == "2026-W46"].iloc[0]
    assert w46["window_days"] == 4
    assert w46["production_hit_rate"] == pytest.approx(0.396)
    assert w46["gate_skipped_reason"] == "challenger_constant"
    assert w46["challenger_constant"] is True
    assert w46["challenger_prediction_std"] == pytest.approx(1.18e-16)


def test_diff_column_is_challenger_minus_production(shadow_dir):
    df = load_shadow_trend(str(shadow_dir))
    diffs = dict(zip(df["week"], df[DIFF_COLUMN]))
    assert math.isnan(diffs["2026-W42"])
    assert diffs["2026-W43"] == pytest.approx(0.05)
    assert diffs["2026-W44"] == pytest.approx(-0.02)
    assert diffs["2026-W46"] == pytest.approx(0.30 - 0.396)


def test_streaks_count_consecutive_weeks(shadow_dir):
    df = load_shadow_trend(str(shadow_dir))
    assert list(df[NOT_WIN_STREAK]) == [0, 0, 1, 2, 3]
    assert list(df[CONSTANT_STREAK]) == [0, 0, 0, 1, 2]


def test_missing_fields_are_empty_and_reset_streaks():
    records = [
        _full("2026-W44", 0.4, 0.3, False, True),
        {"week": "2026-W45"},
        _full("2026-W46", 0.4, 0.3, False, True),
    ]
    df = build_shadow_trend(records)
    old = df.iloc[1]
    assert old["challenger_wins"] is None
    assert old["gate_skipped_reason"] is None
    assert math.isnan(old["production_hit_rate"])
    assert list(df[NOT_WIN_STREAK]) == [1, 0, 1]
    assert list(df[CONSTANT_STREAK]) == [1, 0, 1]


def test_wrong_types_are_empty():
    record = _full("2026-W46", "0.4", True, "no", 1)
    row = build_shadow_trend([record]).iloc[0]
    assert math.isnan(row["production_hit_rate"])
    assert math.isnan(row["challenger_hit_rate"])
    assert row["challenger_wins"] is None
    assert row["challenger_constant"] is None


def test_empty_dir_gives_empty_table(tmp_path):
    df = load_shadow_trend(str(tmp_path))
    assert df.empty
    assert list(df.columns) == SHADOW_COLUMNS
    assert format_shadow_markdown(df) == "データなし"


def test_markdown_leaves_missing_cells_blank(shadow_dir):
    text = format_shadow_markdown(load_shadow_trend(str(shadow_dir)))
    lines = text.splitlines()
    assert lines[0].startswith("| week | written_at |")
    assert len(lines) == 2 + 5
    w42 = lines[2].split("|")[1:-1]
    assert w42[0].strip() == "2026-W42"
    assert w42[1].strip() == ""
    assert "challenger_constant" in lines[-1]
    assert "1.18e-16" in lines[-1]


_OLD_COLUMNS = [
    "week",
    "written_at",
    "window_days",
    "production_hit_rate",
    "challenger_hit_rate",
    "hit_rate_diff",
    "production_sharpe",
    "challenger_sharpe",
    "n_production",
    "n_challenger",
    "challenger_wins",
    "challenger_not_win_streak",
    "challenger_prediction_std",
    "challenger_constant",
    "challenger_constant_streak",
    "gate_skipped_reason",
    "promoted",
    "retry_count",
]
_NEW_COLUMNS = [
    "shadow_evaluated",
    "shadow_failed",
    "gate_evaluated",
    "gate_failed",
    "fresh_challenger_count",
    "shadow_error",
    "production_daily_rank_corr",
    "challenger_daily_rank_corr",
    "win_rule",
]


@pytest.fixture
def status_dir(tmp_path):
    # 新しい欄のそろった週、欄の無い古い週、影の比べが長い文で落ちた週（#2061）
    full = _full("2026-W46", 0.4, 0.3, False, True, "challenger_constant")
    full.update(
        {
            "shadow_evaluated": True,
            "shadow_failed": False,
            "shadow_error": None,
            "fresh_challengers": ["lightgbm", "xgboost"],
            "gate_evaluated": False,
            "gate_failed": False,
            "production_daily_rank_corr": 0.05,
            "challenger_daily_rank_corr": -0.01,
            "win_rule": "daily_rank_corr_v1",
        }
    )
    _write(tmp_path, "2026-W46", full)
    _write(tmp_path, "2026-W45", _full("2026-W45", 0.4, 0.3, False, False))
    _write(
        tmp_path,
        "2026-W47",
        {
            "week": "2026-W47",
            "shadow_evaluated": False,
            "shadow_failed": True,
            "shadow_error": "E" * 500,
            "fresh_challengers": [],
            "gate_evaluated": False,
            "gate_failed": True,
            "gate_skipped_reason": "shadow_failed",
        },
    )
    return tmp_path


def test_new_columns_are_appended_after_old_order():
    assert SHADOW_COLUMNS[: len(_OLD_COLUMNS)] == _OLD_COLUMNS
    assert SHADOW_COLUMNS[len(_OLD_COLUMNS) :] == _NEW_COLUMNS


def test_status_columns_of_full_week(status_dir):
    df = load_shadow_trend(str(status_dir))
    assert list(df.columns) == SHADOW_COLUMNS
    row = df[df["week"] == "2026-W46"].iloc[0]
    assert row["shadow_evaluated"] is True
    assert row["shadow_failed"] is False
    assert row["gate_evaluated"] is False
    assert row["gate_failed"] is False
    assert row["fresh_challenger_count"] == 2
    assert row["shadow_error"] is None
    assert row["production_daily_rank_corr"] == pytest.approx(0.05)
    assert row["challenger_daily_rank_corr"] == pytest.approx(-0.01)
    assert row["win_rule"] == "daily_rank_corr_v1"
    assert row["gate_skipped_reason"] == "challenger_constant"


def test_status_columns_empty_for_old_week(status_dir):
    row = load_shadow_trend(str(status_dir)).iloc[0]
    assert row["week"] == "2026-W45"
    for column in ("shadow_evaluated", "shadow_failed", "gate_evaluated", "gate_failed"):
        assert row[column] is None
    assert math.isnan(row["fresh_challenger_count"])
    assert row["shadow_error"] is None
    assert math.isnan(row["production_daily_rank_corr"])
    assert math.isnan(row["challenger_daily_rank_corr"])
    assert row["win_rule"] is None


def test_long_shadow_error_is_cut(status_dir):
    row = load_shadow_trend(str(status_dir)).iloc[-1]
    assert row["shadow_failed"] is True
    assert row["gate_failed"] is True
    assert row["fresh_challenger_count"] == 0
    assert row["shadow_error"] == "E" * SHADOW_ERROR_HEAD
    assert SHADOW_ERROR_HEAD == 120


def test_wrong_types_of_status_fields_are_empty():
    record = {
        "week": "2026-W46",
        "shadow_failed": "yes",
        "fresh_challengers": "lightgbm",
        "shadow_error": 3,
        "challenger_daily_rank_corr": True,
    }
    row = build_shadow_trend([record]).iloc[0]
    assert row["shadow_failed"] is None
    assert math.isnan(row["fresh_challenger_count"])
    assert row["shadow_error"] is None
    assert math.isnan(row["challenger_daily_rank_corr"])
