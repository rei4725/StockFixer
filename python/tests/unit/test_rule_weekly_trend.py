"""results/rule_accuracy/weekly の週の JSON を読む表のテスト（#1906）。"""

import json
import logging
import math

import pandas as pd
import pytest

from src.reporting.rule_weekly_trend import (
    build_weekly_outcomes,
    build_weekly_rule_best_diff,
    filter_weekly_outcomes,
    format_weekly_markdown,
    load_weekly_reports,
    load_weekly_trend,
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


def _write(dir_path, week, outcomes, diff, n_outcomes, market="jp"):
    report = {
        "iso_week": week,
        "market": market,
        "anchor_horizon": 5,
        "n_outcomes": n_outcomes,
        "outcomes": outcomes,
        "rule_best_diff": diff,
    }
    (dir_path / f"{week}.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def weekly_dir(tmp_path):
    _write(
        tmp_path,
        "2026-W45",
        {"R-001": {"buy": {"1": _stats(10, 0.6, 0.5, 0.01), "5": dict(_EMPTY)}}},
        {
            "evaluated_on": "2026-11-08",
            "previous_evaluated_on": None,
            "added": [{"symbol": "7203", "rule": "R-001"}, {"symbol": "6758", "rule": "R-002"}],
            "removed": [],
            "changed": [],
        },
        10,
    )
    _write(
        tmp_path,
        "2026-W46",
        {
            "R-001": {"buy": {"1": _stats(8, 0.25, 0.5, -0.02)}},
            "R-002": {"sell": {"5": _stats(4, 0.75, 0.5, 0.03)}},
        },
        {
            "evaluated_on": "2026-11-15",
            "previous_evaluated_on": "2026-11-08",
            "added": [{"symbol": "9984", "rule": "R-001"}],
            "removed": [{"symbol": "6758", "rule": "R-002"}],
            "changed": [{"symbol": "7203", "from": "R-001", "to": "R-002"}],
        },
        12,
    )
    return tmp_path


def test_outcome_rows_match_json(weekly_dir):
    df, _ = load_weekly_trend(str(weekly_dir))
    assert list(
        df[["week", "rule", "direction", "horizon"]].itertuples(index=False, name=None)
    ) == [
        ("2026-W45", "R-001", "buy", "1"),
        ("2026-W45", "R-001", "buy", "5"),
        ("2026-W46", "R-001", "buy", "1"),
        ("2026-W46", "R-002", "sell", "5"),
    ]
    row = df.iloc[2]
    assert row["market"] == "jp"
    assert row["n"] == 8
    assert row["hit_rate"] == pytest.approx(0.25)
    assert row["baseline_rate"] == pytest.approx(0.5)
    assert row["edge"] == pytest.approx(-0.25)
    assert row["mean_signed_return"] == pytest.approx(-0.02)
    assert df.iloc[3]["edge"] == pytest.approx(0.25)


def test_diff_rows_match_json(weekly_dir):
    _, diff = load_weekly_trend(str(weekly_dir))
    assert list(diff["week"]) == ["2026-W45", "2026-W46"]
    w45, w46 = diff.iloc[0], diff.iloc[1]
    assert (w45["n_outcomes"], w45["added"], w45["removed"], w45["changed"]) == (10, 2, 0, 0)
    assert pd.isna(w45["previous_evaluated_on"])
    assert (w46["n_outcomes"], w46["added"], w46["removed"], w46["changed"]) == (12, 1, 1, 1)
    assert w46["evaluated_on"] == "2026-11-15"
    assert w46["previous_evaluated_on"] == "2026-11-08"


def test_null_metrics_stay_missing_not_zero(weekly_dir):
    df, _ = load_weekly_trend(str(weekly_dir))
    empty = df.iloc[1]
    assert empty["n"] == 0
    for metric in ("hit_rate", "baseline_rate", "edge", "mean_signed_return"):
        assert math.isnan(empty[metric])


def test_missing_keys_stay_missing():
    reports = [
        {
            "_file_week": "2026-W47",
            "market": "jp",
            "outcomes": {"R-003": {"buy": {"1": {"hit_rate": 0.5}}}},
            "rule_best_diff": {"added": [], "removed": None},
        }
    ]
    df = build_weekly_outcomes(reports)
    assert df.iloc[0]["hit_rate"] == pytest.approx(0.5)
    assert math.isnan(df.iloc[0]["n"])
    assert math.isnan(df.iloc[0]["edge"])
    diff = build_weekly_rule_best_diff(reports)
    row = diff.iloc[0]
    assert math.isnan(row["n_outcomes"])
    assert row["added"] == 0
    assert math.isnan(row["removed"])
    assert math.isnan(row["changed"])
    assert pd.isna(row["evaluated_on"])


def test_missing_diff_block_stays_missing():
    diff = build_weekly_rule_best_diff([{"_file_week": "2026-W47", "market": "jp"}])
    assert all(math.isnan(diff.iloc[0][c]) for c in ("n_outcomes", "added", "removed", "changed"))


def test_broken_json_is_skipped_with_warning(weekly_dir, caplog):
    (weekly_dir / "2026-W47.json").write_text("{broken", encoding="utf-8")
    (weekly_dir / "2026-W48.json").write_text("[1, 2]", encoding="utf-8")
    (weekly_dir / "2026-11-15.json").write_text("{}", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        reports = load_weekly_reports(str(weekly_dir))
    assert [r["_file_week"] for r in reports] == ["2026-W45", "2026-W46"]
    assert "2026-W47.json" in caplog.text
    assert "2026-W48.json" in caplog.text


def test_market_filter_and_missing_market(weekly_dir, caplog):
    _write(weekly_dir, "2026-W47", {"R-009": {"buy": {"1": dict(_EMPTY)}}}, {}, 1, market="us")
    reports = load_weekly_reports(str(weekly_dir)) + [{"_file_week": "2026-W49"}]
    with caplog.at_level(logging.WARNING):
        df = build_weekly_outcomes(reports, market="jp")
        diff = build_weekly_rule_best_diff(reports, market="jp")
    assert set(df["week"]) == {"2026-W45", "2026-W46"}
    assert list(diff["week"]) == ["2026-W45", "2026-W46"]
    assert "2026-W49" in caplog.text
    both = build_weekly_rule_best_diff(load_weekly_reports(str(weekly_dir)))
    assert list(both["market"]) == ["jp", "jp", "us"]


def test_filter_and_empty(weekly_dir, tmp_path_factory):
    df, _ = load_weekly_trend(str(weekly_dir))
    only = filter_weekly_outcomes(df, rule="R-001", horizon="1", direction="buy")
    assert list(only["week"]) == ["2026-W45", "2026-W46"]
    assert filter_weekly_outcomes(df).equals(df)
    empty_out, empty_diff = load_weekly_trend(str(tmp_path_factory.mktemp("none")))
    assert empty_out.empty and empty_diff.empty
    assert format_weekly_markdown(empty_out) == "データなし"


def test_markdown_shows_missing_as_blank(weekly_dir):
    df, diff = load_weekly_trend(str(weekly_dir))
    lines = format_weekly_markdown(df).splitlines()
    assert lines[0].startswith("| week | market | rule |")
    assert lines[3] == "| 2026-W45 | jp | R-001 | buy | 5 | 0 |  |  |  |  |"
    assert (
        lines[2] == "| 2026-W45 | jp | R-001 | buy | 1 | 10 | 0.6000 | 0.5000 | 0.1000 | 0.0100 |"
    )
    diff_lines = format_weekly_markdown(diff).splitlines()
    assert diff_lines[2] == "| 2026-W45 | jp | 10 | 2 | 0 | 0 | 2026-11-08 |  |"
