"""results/rule_accuracy の日ごとの JSON を読む表のテスト（#1837）。"""

import json
import logging
import math

import pytest

from src.reporting.rule_accuracy_trend import (
    build_rule_accuracy_trend,
    filter_rule_trend,
    format_rule_trend_markdown,
    load_rule_accuracy_reports,
    load_rule_accuracy_trend,
)

_EMPTY = {"n": 0, "hit_rate": None, "baseline_rate": None, "edge": None, "mean_signed_return": None}


def _stats(n, hit, base, ret):
    return {
        "n": n,
        "hit_rate": hit,
        "baseline_rate": base,
        "edge": hit - base,
        "mean_signed_return": ret,
    }


def _rule(buy1, sell1=None):
    return {
        "buy": {"1": buy1, "5": dict(_EMPTY), "20": dict(_EMPTY)},
        "sell": {"1": sell1 or dict(_EMPTY), "5": dict(_EMPTY), "20": dict(_EMPTY)},
    }


def _write(dir_path, day, rules_28d, rules_all, market="jp"):
    report = {
        "date": day,
        "market": market,
        "windows": {
            "28d": {"rules": rules_28d, "overall": {}, "predictions": {}},
            "all": {"rules": rules_all, "overall": {}, "predictions": {}},
        },
    }
    (dir_path / f"{day}.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def rule_dir(tmp_path):
    _write(
        tmp_path,
        "2026-11-05",
        {"R-001": _rule(_stats(10, 0.6, 0.5, 0.01))},
        {"R-001": _rule(_stats(20, 0.55, 0.5, 0.005))},
    )
    _write(
        tmp_path,
        "2026-11-06",
        {
            "R-001": _rule(_stats(12, 0.5, 0.5, 0.0)),
            "R-002": _rule(_stats(4, 0.75, 0.5, 0.02)),
        },
        {
            "R-001": _rule(_stats(22, 0.5, 0.5, 0.0)),
            "R-002": _rule(_stats(4, 0.75, 0.5, 0.02)),
        },
    )
    _write(
        tmp_path,
        "2026-11-07",
        {"R-001": _rule(_stats(11, 0.4, 0.5, -0.01))},
        {"R-001": _rule(_stats(23, 0.45, 0.5, -0.002))},
    )
    return tmp_path


def test_table_has_one_row_per_day_window_rule_direction_horizon(rule_dir):
    df = load_rule_accuracy_trend(str(rule_dir))
    # R-001 は3日 × 2窓 × 2向き × 3ホライズン、R-002 は1日だけ
    assert len(df) == 3 * 2 * 2 * 3 + 1 * 2 * 2 * 3
    assert list(df.columns[:6]) == ["date", "market", "window", "rule", "direction", "horizon"]
    r2 = df[df["rule"] == "R-002"]
    assert set(r2["date"]) == {"2026-11-06"}


def test_values_and_edge_trend_in_28d_window(rule_dir):
    df = filter_rule_trend(
        load_rule_accuracy_trend(str(rule_dir)),
        rule="R-001",
        window="28d",
        horizon="1",
        direction="buy",
    )
    assert list(df["date"]) == ["2026-11-05", "2026-11-06", "2026-11-07"]
    assert list(df["n"]) == [10.0, 12.0, 11.0]
    assert df["edge"].tolist() == pytest.approx([0.1, 0.0, -0.1])
    assert df["mean_signed_return"].tolist() == pytest.approx([0.01, 0.0, -0.01])


def test_null_is_kept_apart_from_zero(rule_dir):
    df = load_rule_accuracy_trend(str(rule_dir))
    empty = filter_rule_trend(df, rule="R-001", window="all", horizon="5", direction="buy")
    assert (empty["n"] == 0).all()
    assert all(math.isnan(v) for v in empty["edge"])
    zero = filter_rule_trend(df, rule="R-001", window="all", horizon="1", direction="buy")
    assert zero["edge"].iloc[1] == 0.0

    md = format_rule_trend_markdown(empty.head(1))
    last = md.splitlines()[-1]
    assert last.endswith("| 0.0000 |  |  |  |  |")
    assert "| 0.0000 | 0.0000 |" in format_rule_trend_markdown(zero.iloc[[1]])


def test_csv_leaves_null_cells_empty(rule_dir):
    df = filter_rule_trend(
        load_rule_accuracy_trend(str(rule_dir)), rule="R-001", window="28d", horizon="20"
    )
    line = df.to_csv(index=False).splitlines()[1]
    assert line.endswith(",0.0,,,,")


def test_unreadable_and_other_market_json_are_skipped_with_warning(rule_dir, caplog):
    (rule_dir / "2026-11-08.json").write_text("{broken", encoding="utf-8")
    (rule_dir / "2026-11-09.json").write_text("[1, 2]", encoding="utf-8")
    _write(rule_dir, "2026-11-10", {"R-009": _rule(_stats(5, 0.6, 0.5, 0.0))}, {}, market="us")
    with caplog.at_level(logging.WARNING):
        reports = load_rule_accuracy_reports(str(rule_dir))
        df = build_rule_accuracy_trend(reports, market="jp")
    assert len(reports) == 4
    assert "R-009" not in set(df["rule"])
    assert set(df["market"]) == {"jp"}
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "2026-11-08.json" in messages
    assert "2026-11-09.json" in messages
    assert "us" in messages


def test_without_market_rows_are_split_by_market_column(rule_dir):
    _write(rule_dir, "2026-11-10", {"R-009": _rule(_stats(5, 0.6, 0.5, 0.0))}, {}, market="us")
    df = load_rule_accuracy_trend(str(rule_dir))
    assert set(df[df["rule"] == "R-009"]["market"]) == {"us"}


def test_weekly_and_non_day_files_are_not_read(rule_dir):
    weekly = rule_dir / "weekly"
    weekly.mkdir()
    _write(weekly, "2026-11-09", {"R-W": _rule(_stats(5, 0.6, 0.5, 0.0))}, {})
    _write(rule_dir, "latest", {"R-X": _rule(_stats(5, 0.6, 0.5, 0.0))}, {})
    df = load_rule_accuracy_trend(str(rule_dir))
    assert not {"R-W", "R-X"} & set(df["rule"])


def test_empty_dir_gives_empty_table(tmp_path):
    df = load_rule_accuracy_trend(str(tmp_path))
    assert df.empty
    assert format_rule_trend_markdown(df) == "データなし"
