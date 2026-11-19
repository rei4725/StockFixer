"""reporting/promotion_gate_trend の単体テスト（#2045）。"""

import json
from unittest.mock import patch

import pandas as pd
import pytest

from src.reporting.promotion_gate_trend import (
    build_promotion_gate_trend,
    format_promotion_gate_csv,
    format_promotion_gate_markdown,
    load_promotion_gate_records,
    load_promotion_gate_trend,
)


def _criteria(**passed_overrides):
    criteria = {
        "net_return": {"shadow": 0.12, "current": 0.10, "passed": True},
        "mdd": {"shadow_abs": 0.05, "current_abs": 0.08, "passed": True},
        "sharpe": {"shadow": 1.2, "current": 1.0, "threshold": 0.95, "passed": True},
        "hit_rate": {"shadow": 0.55, "floor": 0.5, "passed": True},
        "slippage": {"shadow": 0.010, "current": 0.012, "passed": True},
        "prediction_spread": {"shadow": 0.02, "threshold": 0.001, "passed": True},
    }
    for name, passed in passed_overrides.items():
        criteria[name]["passed"] = passed
    return criteria


def _result(evaluated_at, eligible, criteria):
    return {
        "shadow_model": "challenger",
        "current_model": "champion",
        "evaluated_at": evaluated_at,
        "eligible": eligible,
        "require_manual_approval": False,
        "criteria": criteria,
        "reason": "",
        "run_ids": [],
        "wf_source_file": "wf_summary.csv",
        "baseline_metrics": {},
    }


def _write(directory, name, payload):
    path = directory / name
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def promotion_dir(tmp_path):
    # わざと日時の逆の順で書く
    _write(
        tmp_path,
        "20261116_030000_promotion_result.json",
        _result(
            "2026-11-16T03:00:00",
            True,
            _criteria(slippage=False, net_return=None, mdd=None, sharpe=None),
        ),
    )
    _write(tmp_path, "20261112_030000_promotion_result.json", "{ broken")
    _write(
        tmp_path,
        "20261109_030000_promotion_result.json",
        _result("2026-11-09T03:00:00", False, _criteria(slippage=False)),
    )
    _write(
        tmp_path,
        "20261102_030000_promotion_result.json",
        _result("2026-11-02T03:00:00", True, _criteria()),
    )
    _write(tmp_path, "20261123_030000_promotion_result.json", json.dumps([1, 2]))
    _write(tmp_path, "notes.json", {"x": 1})
    return tmp_path


def test_rows_order_and_skips(promotion_dir):
    df = load_promotion_gate_trend(str(promotion_dir))
    assert len(df) == 3
    assert list(df["file"]) == [
        "20261102_030000_promotion_result",
        "20261109_030000_promotion_result",
        "20261116_030000_promotion_result",
    ]
    assert list(df["evaluated_at"]) == [
        "2026-11-02T03:00:00",
        "2026-11-09T03:00:00",
        "2026-11-16T03:00:00",
    ]


def test_passed_true_false_and_missing(promotion_dir):
    df = load_promotion_gate_trend(str(promotion_dir))
    assert df["slippage_passed"].tolist() == [True, False, False]
    assert df["net_return_passed"].isna().tolist() == [False, False, True]
    assert df["sharpe_passed"].isna().tolist() == [False, False, True]
    assert df["hit_rate_passed"].tolist() == [True, True, True]
    assert str(df["slippage_passed"].dtype) == "boolean"


def test_values_and_thresholds(promotion_dir):
    df = load_promotion_gate_trend(str(promotion_dir))
    first = df.iloc[0]
    assert first["mdd_shadow"] == pytest.approx(0.05)
    assert first["mdd_current"] == pytest.approx(0.08)
    assert first["hit_rate_threshold"] == pytest.approx(0.5)
    assert pd.isna(first["hit_rate_current"])
    assert first["sharpe_threshold"] == pytest.approx(0.95)
    assert pd.isna(first["slippage_threshold"])
    assert first["slippage_shadow"] == pytest.approx(0.010)
    assert first["wf_source_file"] == "wf_summary.csv"
    assert first["shadow_model"] == "challenger"
    assert first["current_model"] == "champion"
    assert df["eligible"].tolist() == [True, False, True]


def test_fail_streaks(promotion_dir):
    df = load_promotion_gate_trend(str(promotion_dir))
    assert df["slippage_fail_streak"].tolist() == [0, 1, 2]
    assert df["hit_rate_fail_streak"].tolist() == [0, 0, 0]
    assert df["net_return_fail_streak"].isna().tolist() == [False, False, True]
    assert str(df["slippage_fail_streak"].dtype) == "Int64"


def test_missing_passed_carries_streak_over():
    records = [
        dict(_result("a", False, _criteria(slippage=False)), _file="1"),
        dict(_result("b", True, _criteria(slippage=None)), _file="2"),
        dict(_result("c", False, _criteria(slippage=False)), _file="3"),
        dict(_result("d", True, _criteria()), _file="4"),
    ]
    del records[1]["criteria"]["slippage"]["passed"]
    df = build_promotion_gate_trend(records)
    assert df["slippage_fail_streak"].isna().tolist() == [False, True, False, False]
    assert df["slippage_fail_streak"].dropna().tolist() == [1, 2, 0]


def test_broken_and_wrong_shape_are_skipped_with_warning(promotion_dir):
    with patch("src.reporting.promotion_gate_trend.logger") as mock_logger:
        records = load_promotion_gate_records(str(promotion_dir))
    assert len(records) == 3
    assert mock_logger.warning.call_count == 2


def test_criteria_wrong_shape_skipped(tmp_path):
    bad = _result("x", False, [])
    _write(tmp_path, "20261101_000000_promotion_result.json", bad)
    assert load_promotion_gate_records(str(tmp_path)) == []


def test_missing_dir_returns_empty(tmp_path):
    df = load_promotion_gate_trend(str(tmp_path / "none"))
    assert df.empty
    assert format_promotion_gate_markdown(df) == "データなし"


def test_markdown_and_csv(promotion_dir):
    df = load_promotion_gate_trend(str(promotion_dir))
    md = format_promotion_gate_markdown(df)
    lines = md.splitlines()
    assert len(lines) == 2 + 3
    assert "slippage_fail_streak" in lines[0]
    assert "null" in lines[-1]
    csv = format_promotion_gate_csv(df)
    csv_lines = csv.strip().splitlines()
    assert len(csv_lines) == 4
    assert csv_lines[0].startswith("file,evaluated_at,shadow_model,current_model,eligible")
    assert "null" in csv_lines[-1]
