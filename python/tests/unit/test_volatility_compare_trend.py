"""results/backtest_compare/volatility_sizing の週ごとの JSON を読む表のテスト（#1864）。"""

import json
import math

import pytest

from src.reporting.volatility_compare_trend import (
    TREND_COLUMNS,
    build_compare_trend,
    format_compare_markdown,
    load_compare_reports,
    load_compare_trend,
)


def _form(name, k, total, dd, rov, trades=10, stops=2):
    return {
        "form": name,
        "k": k,
        "symbols": 3,
        "num_trades": trades,
        "stop_outs": stops,
        "max_drawdown": dd,
        "total_return": total,
        "return_over_volatility": rov,
    }


def _market(forms, start="2021-11-01", end="2026-11-06", symbols=3, failed=1):
    return {
        "symbols_target": symbols + failed,
        "symbols_over_limit": 0,
        "symbols": symbols,
        "symbols_failed": failed,
        "symbols_lacking_forecast": 0,
        "period": {"start": start, "end": end},
        "forms": forms,
    }


def _write(dir_path, week, markets):
    report = {"week": week, "as_of": "2026-11-08", "start_date": "x", "end_date": "y"}
    report["markets"] = markets
    (dir_path / f"{week}.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def compare_dir(tmp_path):
    _write(
        tmp_path,
        "2026-W45",
        {
            "us": _market(
                [
                    _form("fixed", None, -0.4, -0.5, 0.1),
                    _form("size", None, -0.3, -0.45, 0.2),
                    _form("stop", 2.0, -0.2, -0.3, None),
                ]
            )
        },
    )
    _write(
        tmp_path,
        "2026-W46",
        {
            "us": _market(
                [_form("fixed", None, 0.1, -0.2, 0.5), _form("both", 3.0, 0.3, -0.1, 0.4)]
            ),
            # fixed の欠けた市場
            "jp": _market([_form("stop", 4.0, 0.05, -0.1, 0.3)], symbols=2, failed=0),
        },
    )
    (tmp_path / "2026-W47.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "2026-W48.json").write_text("[1, 2]", encoding="utf-8")
    return tmp_path


def test_load_skips_broken_files(compare_dir):
    reports = load_compare_reports(str(compare_dir))
    assert [r["week"] for r in reports] == ["2026-W45", "2026-W46"]


def test_rows_per_week_market_form_k(compare_dir):
    df = load_compare_trend(str(compare_dir))
    assert list(df.columns) == TREND_COLUMNS
    keys = [(r.week, r.market, r.form) for r in df.itertuples()]
    assert keys == [
        ("2026-W45", "us", "fixed"),
        ("2026-W45", "us", "size"),
        ("2026-W45", "us", "stop"),
        ("2026-W46", "jp", "stop"),
        ("2026-W46", "us", "fixed"),
        ("2026-W46", "us", "both"),
    ]
    stop = df.iloc[2]
    assert stop["k"] == 2.0
    assert math.isnan(df.iloc[0]["k"])
    assert stop["num_trades"] == 10.0
    assert stop["stop_outs"] == 2.0
    assert stop["period_start"] == "2021-11-01"
    assert stop["period_end"] == "2026-11-06"
    assert stop["symbols"] == 3.0
    assert stop["symbols_failed"] == 1.0


def test_diff_against_fixed_same_week_and_market(compare_dir):
    df = load_compare_trend(str(compare_dir))
    size = df.iloc[1]
    assert size["total_return_vs_fixed"] == pytest.approx(0.1)
    assert size["max_drawdown_vs_fixed"] == pytest.approx(0.05)
    assert size["return_over_volatility_vs_fixed"] == pytest.approx(0.1)
    fixed = df.iloc[0]
    assert fixed["total_return_vs_fixed"] == pytest.approx(0.0)
    # 値の無い物差しの差は空
    assert math.isnan(df.iloc[2]["return_over_volatility_vs_fixed"])
    both = df.iloc[5]
    assert both["total_return_vs_fixed"] == pytest.approx(0.2)
    assert both["return_over_volatility_vs_fixed"] == pytest.approx(-0.1)


def test_diff_empty_when_fixed_missing(compare_dir):
    df = load_compare_trend(str(compare_dir))
    jp = df.iloc[3]
    assert jp["total_return"] == pytest.approx(0.05)
    assert jp["symbols_failed"] == 0.0
    for column in ("total_return_vs_fixed", "max_drawdown_vs_fixed"):
        assert math.isnan(jp[column])
    assert math.isnan(jp["return_over_volatility_vs_fixed"])


def test_report_without_markets_is_skipped():
    df = build_compare_trend([{"week": "2026-W01"}, {"week": "2026-W02", "markets": {"us": {}}}])
    assert df.empty
    assert list(df.columns) == TREND_COLUMNS


def test_markdown(compare_dir):
    df = load_compare_trend(str(compare_dir))
    text = format_compare_markdown(df)
    lines = text.splitlines()
    assert lines[0].startswith("| week | market | form | k |")
    assert len(lines) == 2 + len(df)
    assert "| 2026-W46 | jp | stop | 4.0000 |" in text
    assert format_compare_markdown(df.iloc[0:0]) == "データなし"
