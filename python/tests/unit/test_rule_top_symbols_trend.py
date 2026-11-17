"""rule_top_symbols_trend のテスト（#1972）。"""

import json
import math
import os
import sys

import pandas as pd

from src.reporting.rule_top_symbols_trend import (
    TREND_COLUMNS,
    build_rule_top_symbols_trend,
    build_top_symbol_rows,
    load_rule_top_symbols_trend,
)


def _sym(symbol, rule, direction, n, hit, edge):
    return {
        "symbol": symbol,
        "rule_name": rule,
        "direction": direction,
        "n_h5": n,
        "hit_rate_h5": hit,
        "edge_h5": edge,
    }


def _report(day, top_symbols, market="jp"):
    return {"date": day, "market": market, "top_symbols": top_symbols}


def _write(tmp_path, day, report):
    with open(os.path.join(tmp_path, f"{day}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f)


def _reports():
    return [
        dict(
            _report(
                "2026-11-05",
                [
                    _sym("7752", "bollinger_band", "sell", 76, 0.30, -0.158),
                    _sym("6758", "rsi", "buy", 40, 0.55, 0.05),
                ],
            ),
            _file_date="2026-11-05",
        ),
        dict(
            _report(
                "2026-11-06",
                [
                    _sym("7752", "bollinger_band", "sell", 77, 0.31, -0.170),
                    _sym("7752", "bollinger_band", "buy", 10, 0.6, 0.1),
                ],
            ),
            _file_date="2026-11-06",
        ),
        dict(
            _report(
                "2026-11-07",
                [_sym("7752", "bollinger_band", "sell", 79, 0.33, -0.134)],
            ),
            _file_date="2026-11-07",
        ),
    ]


def test_same_group_collapses_across_days_and_takes_last_day_values():
    df = build_rule_top_symbols_trend(_reports())
    assert list(df.columns) == TREND_COLUMNS
    row = df[
        (df["symbol"] == "7752")
        & (df["rule_name"] == "bollinger_band")
        & (df["direction"] == "sell")
    ]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["market"] == "jp"
    assert r["n_days"] == 3
    assert r["first_date"] == "2026-11-05"
    assert r["last_date"] == "2026-11-07"
    assert r["last_n_h5"] == 79
    assert math.isclose(r["last_hit_rate_h5"], 0.33)
    assert math.isclose(r["last_edge_h5"], -0.134)
    assert math.isclose(r["min_edge_h5"], -0.170)
    assert math.isclose(r["max_edge_h5"], -0.134)
    # 出た日数の多い順で先頭に来る
    assert df.iloc[0]["symbol"] == "7752" and df.iloc[0]["direction"] == "sell"
    assert len(df) == 3


def test_direction_is_part_of_the_group():
    df = build_rule_top_symbols_trend(_reports())
    buy = df[(df["symbol"] == "7752") & (df["direction"] == "buy")]
    assert len(buy) == 1
    assert buy.iloc[0]["n_days"] == 1


def test_malformed_items_are_skipped_and_null_becomes_nan():
    reports = [
        {
            "market": "jp",
            "_file_date": "2026-11-05",
            "top_symbols": [
                "not a dict",
                {"rule_name": "rsi", "direction": "buy"},
                {"symbol": "", "rule_name": "rsi", "direction": "buy"},
                _sym("1301", "rsi", "buy", None, "x", None),
            ],
        },
        {"market": "jp", "_file_date": "2026-11-06", "top_symbols": "broken"},
        {"market": "jp", "_file_date": "2026-11-07"},
    ]
    rows = build_top_symbol_rows(reports)
    assert len(rows) == 1
    assert math.isnan(rows.iloc[0]["n_h5"])
    assert math.isnan(rows.iloc[0]["hit_rate_h5"])
    df = build_rule_top_symbols_trend(reports)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["symbol"] == "1301"
    assert math.isnan(r["last_n_h5"])
    assert math.isnan(r["last_edge_h5"])
    assert math.isnan(r["min_edge_h5"])


def test_bool_is_not_a_number():
    reports = [
        {
            "market": "jp",
            "_file_date": "2026-11-05",
            "top_symbols": [_sym("1301", "rsi", "buy", True, 0.5, False)],
        }
    ]
    df = build_rule_top_symbols_trend(reports)
    assert math.isnan(df.iloc[0]["last_n_h5"])
    assert math.isnan(df.iloc[0]["last_edge_h5"])


def test_markets_are_separate_rows():
    reports = [
        dict(_report("2026-11-05", [_sym("AAPL", "rsi", "buy", 5, 0.5, 0.0)], "us")),
        dict(_report("2026-11-05", [_sym("AAPL", "rsi", "buy", 6, 0.5, 0.1)], "jp")),
    ]
    reports[0]["_file_date"] = "2026-11-05"
    reports[1]["_file_date"] = "2026-11-05"
    df = build_rule_top_symbols_trend(reports)
    assert sorted(df["market"]) == ["jp", "us"]


def test_empty_reports_give_empty_frame():
    df = build_rule_top_symbols_trend([])
    assert df.empty
    assert list(df.columns) == TREND_COLUMNS


def test_load_from_directory(tmp_path):
    for report in _reports():
        day = report["_file_date"]
        body = {k: v for k, v in report.items() if k != "_file_date"}
        _write(tmp_path, day, body)
    os.makedirs(os.path.join(tmp_path, "weekly"))
    df = load_rule_top_symbols_trend(str(tmp_path))
    assert df.iloc[0]["n_days"] == 3
    assert isinstance(df, pd.DataFrame)


def test_cli_writes_csv_with_min_days(tmp_path, monkeypatch):
    import run_rule_top_symbols_trend as cli

    for report in _reports():
        day = report["_file_date"]
        _write(tmp_path, day, {k: v for k, v in report.items() if k != "_file_date"})
    out = tmp_path / "out.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_rule_top_symbols_trend.py",
            "--dir",
            str(tmp_path),
            "--min-days",
            "2",
            "--format",
            "csv",
            "--output",
            str(out),
        ],
    )
    cli.main()
    df = pd.read_csv(out, dtype={"symbol": str})
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "7752"
    assert df.iloc[0]["n_days"] == 3


def test_cli_markdown_to_stdout(tmp_path, monkeypatch, capsys):
    import run_rule_top_symbols_trend as cli

    for report in _reports():
        day = report["_file_date"]
        _write(tmp_path, day, {k: v for k, v in report.items() if k != "_file_date"})
    monkeypatch.setattr(sys, "argv", ["run_rule_top_symbols_trend.py", "--dir", str(tmp_path)])
    cli.main()
    text = capsys.readouterr().out
    assert "7752" in text
    assert "bollinger_band" in text
