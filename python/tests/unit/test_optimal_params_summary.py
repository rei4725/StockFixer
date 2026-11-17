"""results/optimal_params.json の市場ごとの集計（#1950）のテスト。

期待値は、下の小さな JSON を手で数えた値で書く。
"""

import json
import sys
from datetime import datetime

import pytest

from src.reporting.optimal_params_summary import (
    SUMMARY_COLUMNS,
    build_param_rows,
    format_summary_markdown,
    load_optimal_params,
    stale_rows,
    summarize_by_market,
)

NOW = datetime(2026, 11, 17, 12, 0, 0)


def _row(market, symbol, ts, threshold, num_trades, stop_loss=0.05, sizing="full"):
    return {
        "market": market,
        "symbol": symbol,
        "timestamp": ts,
        "threshold": threshold,
        "stop_loss_pct": stop_loss,
        "position_sizing": sizing,
        "metrics": {"num_trades": num_trades},
    }


def _params() -> dict:
    return {
        # jp: 3 行（1 行は古い）。num_trades は 0 / 2 / 15
        "jp_1332": _row("jp", "1332", "2026-11-10T09:00:00", 0.01, 2),
        "jp_5202": _row("jp", "5202", "2026-03-23T09:00:00", 0.03, 0, stop_loss=None),
        "jp_7203": _row("jp", "7203", "2026-11-15T09:00:00", 0.02, 15, sizing="atr"),
        # us: 2 行（2 行とも古い）。num_trades は 5 / 1
        "us_AVB": _row("us", "AVB", "2026-03-23T09:00:00", 0.005, 5, stop_loss=None),
        "us_BK": _row("us", "BK", "2026-10-01T09:00:00", 0.015, 1, stop_loss=None),
        # 形の違う行（飛ばす）
        "bad_list": [1, 2, 3],
        "bad_ts": _row("us", "X", "not-a-date", 0.01, 3),
        "bad_threshold": _row("us", "Y", "2026-11-16T09:00:00", "high", 3),
        "bad_metrics": {**_row("jp", "Z", "2026-11-16T09:00:00", 0.01, 3), "metrics": None},
        "bad_market": _row(None, "W", "2026-11-16T09:00:00", 0.01, 3),
    }


def test_build_param_rows_skips_malformed_rows():
    rows = build_param_rows(_params())
    assert sorted(rows["key"]) == ["jp_1332", "jp_5202", "jp_7203", "us_AVB", "us_BK"]
    assert len(rows) == 5


def test_summarize_counts_per_market():
    summary = summarize_by_market(build_param_rows(_params()), now=NOW, stale_days=14)
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert list(summary["market"]) == ["jp", "us"]
    jp = summary.iloc[0].to_dict()
    us = summary.iloc[1].to_dict()

    assert jp["rows"] == 3
    assert jp["oldest"] == "2026-03-23"
    assert jp["newest"] == "2026-11-15"
    assert jp["stale"] == 1
    assert (jp["trades_0"], jp["trades_1_2"], jp["trades_3_9"], jp["trades_10_plus"]) == (
        1,
        1,
        0,
        1,
    )
    assert jp["threshold_min"] == pytest.approx(0.01)
    assert jp["threshold_median"] == pytest.approx(0.02)
    assert jp["threshold_max"] == pytest.approx(0.03)
    assert jp["stop_loss_null"] == 1
    assert jp["position_sizing"] == "atr:1;full:2"

    assert us["rows"] == 2
    assert us["oldest"] == "2026-03-23"
    assert us["newest"] == "2026-10-01"
    assert us["stale"] == 2
    assert (us["trades_0"], us["trades_1_2"], us["trades_3_9"], us["trades_10_plus"]) == (
        0,
        1,
        1,
        0,
    )
    assert us["threshold_median"] == pytest.approx(0.01)
    assert us["stop_loss_null"] == 2
    assert us["position_sizing"] == "full:2"


def test_stale_days_changes_stale_count():
    rows = build_param_rows(_params())
    # 2026-11-10 は 7 日前。N=5 なら jp は 1332 と 5202 の 2 行が古い
    summary = summarize_by_market(rows, now=NOW, stale_days=5)
    assert int(summary.iloc[0]["stale"]) == 2


def test_stale_rows_lists_old_symbols_oldest_first():
    stale = stale_rows(build_param_rows(_params()), now=NOW, stale_days=14)
    assert list(stale["key"]) == ["jp_5202", "us_AVB", "us_BK"]
    assert list(stale["age_days"]) == [239, 239, 47]


def test_empty_params_gives_empty_summary():
    summary = summarize_by_market(build_param_rows({}), now=NOW)
    assert summary.empty
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert format_summary_markdown(summary) == "データなし"


def test_load_optimal_params_missing_file_returns_none(tmp_path):
    assert load_optimal_params(str(tmp_path / "none.json")) is None


def test_load_optimal_params_broken_or_wrong_shape_returns_none(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_optimal_params(str(broken)) is None
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    assert load_optimal_params(str(listed)) is None


def test_load_optimal_params_reads_dict(tmp_path):
    path = tmp_path / "optimal_params.json"
    path.write_text(json.dumps(_params()), encoding="utf-8")
    loaded = load_optimal_params(str(path))
    assert loaded is not None
    assert "jp_1332" in loaded


def test_format_summary_markdown_has_header_and_rows():
    summary = summarize_by_market(build_param_rows(_params()), now=NOW)
    text = format_summary_markdown(summary)
    lines = text.splitlines()
    assert lines[0].startswith("| market | rows |")
    assert len(lines) == 4
    assert "atr:1;full:2" in lines[2]


def test_run_script_writes_nothing_when_file_missing(tmp_path, monkeypatch, capsys):
    import run_optimal_params_summary

    out = tmp_path / "out.md"
    argv = ["prog", "--path", str(tmp_path / "missing.json"), "--output", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    run_optimal_params_summary.main()
    assert capsys.readouterr().out == ""
    assert not out.exists()


def test_run_script_prints_summary_and_stale(tmp_path, monkeypatch, capsys):
    import run_optimal_params_summary

    path = tmp_path / "optimal_params.json"
    path.write_text(json.dumps(_params()), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["prog", "--path", str(path), "--stale"])
    run_optimal_params_summary.main()
    text = capsys.readouterr().out
    assert "| market | rows |" in text
    assert "us_AVB" in text
