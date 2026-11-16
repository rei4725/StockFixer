"""results/rule_accuracy の日ごとの JSON の backtest_gap を読む表のテスト（#1891）。"""

import json
import logging
import math

import pytest

from src.reporting.rule_backtest_gap_trend import (
    GAP_METRICS,
    build_rule_backtest_gap_trend,
    filter_gap_trend,
    format_gap_trend_markdown,
    load_rule_backtest_gap_trend,
)


def _entry(win, hit, n, n_un, approx, **extra):
    out = {
        "backtest_win_rate": win,
        "live_hit_rate_h5": hit,
        "n_h5": n,
        "n_unmatched": n_un,
        "approx": approx,
    }
    out.update(extra)
    return out


def _write(dir_path, day, body, market="jp"):
    report = {"date": day, "market": market, "windows": {}}
    report.update(body)
    (dir_path / f"{day}.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def gap_dir(tmp_path):
    # 古い形: backtest_gap がルールごとに向きをまとめた1段
    _write(
        tmp_path,
        "2026-11-13",
        {"backtest_gap": {"R-001": _entry(0.74, 0.54, 100, 20, False)}},
    )
    # 新しい形: backtest_gap が buy・sell に分かれ、まとめた形は backtest_gap_combined
    _write(
        tmp_path,
        "2026-11-16",
        {
            "backtest_gap": {
                "R-001": {
                    "buy": _entry(0.76, 0.6, 40, 5, False, baseline_rate=0.5, edge=0.1),
                    "sell": _entry(
                        None, 0.0, 0, 0, True, baseline_rate=None, edge=None, n_sell_only=3
                    ),
                }
            },
            "backtest_gap_combined": {"R-001": _entry(0.75, 0.55, 110, 22, True)},
        },
    )
    # 壊れた JSON は飛ばす
    (tmp_path / "2026-11-14.json").write_text("{not json", encoding="utf-8")
    return tmp_path


def test_old_and_new_days_share_columns(gap_dir, caplog):
    with caplog.at_level(logging.WARNING):
        df = load_rule_backtest_gap_trend(str(gap_dir))
    assert list(df.columns) == ["date", "market", "rule", "direction"] + list(GAP_METRICS)
    assert list(zip(df["date"], df["direction"])) == [
        ("2026-11-13", "combined"),
        ("2026-11-16", "buy"),
        ("2026-11-16", "combined"),
        ("2026-11-16", "sell"),
    ]
    assert "2026-11-14.json" in caplog.text

    old = df.iloc[0]
    assert old["backtest_win_rate"] == pytest.approx(0.74)
    assert old["gap"] == pytest.approx(0.20)
    assert old["n_h5"] == 100
    assert old["approx"] is False
    # 古い形には baseline_rate・edge・n_sell_only の欄が無いので空
    assert math.isnan(old["baseline_rate"])
    assert math.isnan(old["edge"])
    assert math.isnan(old["n_sell_only"])

    buy = df.iloc[1]
    assert buy["edge"] == pytest.approx(0.1)
    assert buy["gap"] == pytest.approx(0.16)
    combined = df.iloc[2]
    assert combined["n_unmatched"] == 22
    assert combined["approx"] is True


def test_null_is_kept_apart_from_zero(gap_dir):
    df = load_rule_backtest_gap_trend(str(gap_dir))
    sell = filter_gap_trend(df, rule="R-001", direction="sell").iloc[0]
    assert math.isnan(sell["backtest_win_rate"])
    assert math.isnan(sell["gap"])
    assert math.isnan(sell["edge"])
    assert sell["live_hit_rate_h5"] == 0.0
    assert sell["n_h5"] == 0.0
    assert sell["n_sell_only"] == 3

    md = format_gap_trend_markdown(filter_gap_trend(df, direction="sell"))
    row = md.splitlines()[2]
    # backtest_win_rate は空、live_hit_rate_h5 は 0.0000
    assert row.startswith("| 2026-11-16 | jp | R-001 | sell |  | 0.0000 |  | 0.0000 |")
    assert "| true |" in row


def test_market_filter_and_missing_section(tmp_path, caplog):
    _write(tmp_path, "2026-11-10", {"backtest_gap": {"R-001": _entry(0.7, 0.5, 1, 0, False)}})
    _write(
        tmp_path,
        "2026-11-11",
        {"backtest_gap": {"R-001": _entry(0.7, 0.5, 1, 0, False)}},
        market="us",
    )
    _write(tmp_path, "2026-11-12", {})
    (tmp_path / "2026-11-15.json").write_text("[1, 2]", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        df = load_rule_backtest_gap_trend(str(tmp_path), market="jp")
    assert list(df["date"]) == ["2026-11-10"]
    assert "backtest_gap" in caplog.text


def test_report_without_market_and_odd_entries_are_skipped():
    reports = [
        {"_file_date": "2026-11-01", "backtest_gap": {"R-001": _entry(0.7, 0.5, 1, 0, False)}},
        {
            "_file_date": "2026-11-02",
            "market": "jp",
            "backtest_gap": {"R-001": "x", "R-002": {"buy": "x", "sell": {"n_h5": 2}}},
            "backtest_gap_combined": {"R-001": [1], "R-002": {"approx": "yes"}},
        },
    ]
    df = build_rule_backtest_gap_trend(reports)
    assert list(zip(df["rule"], df["direction"])) == [("R-002", "combined"), ("R-002", "sell")]
    assert df.iloc[0]["approx"] is None


def test_empty_table_formats_as_no_data(tmp_path):
    df = load_rule_backtest_gap_trend(str(tmp_path))
    assert df.empty
    assert list(df.columns) == ["date", "market", "rule", "direction"] + list(GAP_METRICS)
    assert format_gap_trend_markdown(df) == "データなし"
