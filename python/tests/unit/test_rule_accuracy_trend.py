"""results/rule_accuracy の日ごとの JSON を読む表のテスト（#1837）。"""

import json
import logging
import math

import pytest

from src.reporting.rule_accuracy_trend import (
    build_rule_accuracy_summary_trend,
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


# --- overall と predictions の節（#1959） ---

_NULL = {"n": 0, "hit_rate": None, "baseline_rate": None, "edge": None, "mean_signed_return": None}


def _summary_report(day, market="jp"):
    """手で作った overall と predictions の節を持つ1日の JSON。"""
    return {
        "_file_date": day,
        "market": market,
        "windows": {
            "28d": {
                "rules": {"R-001": _rule(_stats(10, 0.6, 0.5, 0.01))},
                "overall": {
                    "buy": {
                        "1": {
                            "n": 40,
                            "hit_rate": 0.55,
                            "baseline_rate": 0.5,
                            "edge": 0.05,
                            "mean_signed_return": 0.002,
                        },
                        "5": dict(_NULL),
                    },
                    "sell": {
                        "1": {
                            "n": 8,
                            "hit_rate": 0.5,
                            "baseline_rate": 0.5,
                            "edge": 0.0,
                            "mean_signed_return": 0.0,
                        },
                    },
                },
                "predictions": {
                    "production": {
                        "h1_claim": {
                            "1": {
                                "n": 4311,
                                "hit_rate": 0.5115,
                                "baseline_rate": 0.51,
                                "edge": 0.0015,
                                "mean_signed_return": 0.0003,
                            },
                        },
                        "h5_claim": {"5": dict(_NULL)},
                    },
                    "challenger": {
                        "h1_claim": {
                            "5": {
                                "n": 2239,
                                "hit_rate": 0.5239,
                                "baseline_rate": 0.52,
                                "edge": 0.0039,
                                "mean_signed_return": -0.0011,
                            },
                        },
                    },
                },
            },
        },
    }


def _one(df, rule, direction, horizon):
    out = filter_rule_trend(df, rule=rule, window="28d", horizon=horizon, direction=direction)
    assert len(out) == 1
    return out.iloc[0]


def test_summary_rows_carry_overall_and_prediction_values():
    df = build_rule_accuracy_summary_trend([_summary_report("2026-11-17")])
    assert list(df.columns) == list(build_rule_accuracy_trend([]).columns)
    # overall 3行（buy 1・buy 5・sell 1）、予想 3行（production h1 1・h5 5、challenger h1 5）
    assert len(df) == 6
    assert "R-001" not in set(df["rule"])
    assert set(df["rule"]) == {"overall", "prediction:production", "prediction:challenger"}

    buy1 = _one(df, "overall", "buy", "1")
    assert (buy1["date"], buy1["market"]) == ("2026-11-17", "jp")
    assert buy1["n"] == 40.0
    assert buy1["hit_rate"] == pytest.approx(0.55)
    assert buy1["baseline_rate"] == pytest.approx(0.5)
    assert buy1["edge"] == pytest.approx(0.05)
    assert buy1["mean_signed_return"] == pytest.approx(0.002)

    prod = _one(df, "prediction:production", "h1_claim", "1")
    assert prod["n"] == 4311.0
    assert prod["hit_rate"] == pytest.approx(0.5115)
    assert prod["baseline_rate"] == pytest.approx(0.51)
    assert prod["edge"] == pytest.approx(0.0015)
    assert prod["mean_signed_return"] == pytest.approx(0.0003)

    chal = _one(df, "prediction:challenger", "h1_claim", "5")
    assert chal["n"] == 2239.0
    assert chal["edge"] == pytest.approx(0.0039)
    assert chal["mean_signed_return"] == pytest.approx(-0.0011)


def test_summary_null_is_nan_and_zero_stays_zero():
    df = build_rule_accuracy_summary_trend([_summary_report("2026-11-17")])
    for rule, direction, horizon in (
        ("overall", "buy", "5"),
        ("prediction:production", "h5_claim", "5"),
    ):
        row = _one(df, rule, direction, horizon)
        assert row["n"] == 0.0
        for metric in ("hit_rate", "baseline_rate", "edge", "mean_signed_return"):
            assert math.isnan(row[metric])
    sell1 = _one(df, "overall", "sell", "1")
    assert sell1["edge"] == 0.0
    assert sell1["mean_signed_return"] == 0.0


def test_summary_skips_old_json_without_sections_and_other_market():
    old = {
        "_file_date": "2026-11-05",
        "market": "jp",
        "windows": {"28d": {"rules": {"R-001": _rule(_stats(10, 0.6, 0.5, 0.01))}}},
    }
    other = _summary_report("2026-11-18", market="us")
    df = build_rule_accuracy_summary_trend([old, _summary_report("2026-11-17"), other], "jp")
    assert set(df["date"]) == {"2026-11-17"}
    assert set(df["market"]) == {"jp"}
    assert len(df) == 6
    assert build_rule_accuracy_summary_trend([old]).empty


def test_default_table_does_not_include_summary_rows():
    df = build_rule_accuracy_trend([_summary_report("2026-11-17")])
    assert set(df["rule"]) == {"R-001"}
