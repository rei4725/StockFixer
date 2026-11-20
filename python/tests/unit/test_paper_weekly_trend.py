"""reporting/paper_weekly_trend.py のテスト（#1862）。"""

import json
import math

import pytest

from src.reporting.paper_weekly_trend import (
    CUM_EXCESS_COLUMN,
    CUM_WEEKLY_COLUMN,
    build_paper_weekly_trend,
    format_paper_weekly_markdown,
    load_paper_weekly_reports,
    load_paper_weekly_trend,
)


def _old_market(weekly, bench, excess, fills=1):
    """#1792 より前の形（exposure・excess_basis・excess_return_unscaled が無い）。"""
    return {
        "weekly_return": weekly,
        "benchmark_ticker": "^N225",
        "benchmark_return": bench,
        "excess_return": excess,
        "fills": fills,
        "valued_at_fill_price": [],
        "partial_week": False,
        "missing_week_end_close": [],
    }


def _new_market(weekly, bench, excess, unscaled, exposure, fills=2, partial=False):
    return {
        "weekly_return": weekly,
        "benchmark_ticker": "^N225",
        "benchmark_return": bench,
        "excess_return": excess,
        "excess_return_unscaled": unscaled,
        "exposure": exposure,
        "excess_basis": "exposure_scaled",
        "fills": fills,
        "valued_at_fill_price": [],
        "partial_week": partial,
        "missing_week_end_close": [],
    }


def _write(path, week, start, end, markets):
    path.write_text(
        json.dumps({"week": week, "start": start, "end": end, "markets": markets}),
        encoding="utf-8",
    )


@pytest.fixture
def paper_dir(tmp_path):
    _write(
        tmp_path / "2026-W43.json",
        "2026-W43",
        "2026-10-19",
        "2026-10-23",
        {"jp": _old_market(0.02, 0.01, 0.01), "us": _old_market(0.0, 0.005, -0.005, fills=0)},
    )
    _write(
        tmp_path / "2026-W44.json",
        "2026-W44",
        "2026-10-26",
        "2026-10-30",
        {"jp": _old_market(-0.01, 0.0, -0.01)},
    )
    _write(
        tmp_path / "2026-W45.json",
        "2026-W45",
        "2026-11-02",
        "2026-11-06",
        {
            "jp": _new_market(0.03, 0.02, 0.02, 0.01, 0.5, partial=True),
            "us": _new_market(None, 0.01, None, None, None),
        },
    )
    _write(
        tmp_path / "2026-W46.json",
        "2026-W46",
        "2026-11-09",
        "2026-11-13",
        {"jp": _new_market(0.01, 0.0, 0.01, 0.01, 0.4)},
    )
    (tmp_path / "2026-W42.json").write_text("{壊れた", encoding="utf-8")
    (tmp_path / "2026-W41.json").write_text("[1, 2]", encoding="utf-8")
    return tmp_path


def test_rows_per_week_and_market(paper_dir):
    df = load_paper_weekly_trend(str(paper_dir))
    keys = list(zip(df["week"], df["market"]))
    assert keys == [
        ("2026-W43", "jp"),
        ("2026-W43", "us"),
        ("2026-W44", "jp"),
        ("2026-W45", "jp"),
        ("2026-W45", "us"),
        ("2026-W46", "jp"),
    ]
    row = df[(df["week"] == "2026-W45") & (df["market"] == "jp")].iloc[0]
    assert row["start"] == "2026-11-02"
    assert row["end"] == "2026-11-06"
    assert row["benchmark_ticker"] == "^N225"
    assert row["weekly_return"] == pytest.approx(0.03)
    assert row["excess_return"] == pytest.approx(0.02)
    assert row["excess_return_unscaled"] == pytest.approx(0.01)
    assert row["exposure"] == pytest.approx(0.5)
    assert row["excess_basis"] == "exposure_scaled"
    assert row["fills"] == 2
    assert bool(row["partial_week"]) is True


def test_old_weeks_leave_new_columns_empty(paper_dir):
    df = load_paper_weekly_trend(str(paper_dir))
    old = df[df["week"] == "2026-W43"]
    assert old["exposure"].isna().all()
    assert old["excess_return_unscaled"].isna().all()
    assert old["excess_basis"].isna().all()
    markdown = format_paper_weekly_markdown(old)
    jp_line = [line for line in markdown.splitlines() if "| jp |" in line][0]
    cells = [c.strip() for c in jp_line.strip("|").split("|")]
    columns = [c.strip() for c in markdown.splitlines()[0].strip("|").split("|")]
    assert cells[columns.index("exposure")] == ""
    assert cells[columns.index("excess_basis")] == ""
    assert cells[columns.index("excess_return")] == "0.010000"


def test_cumulative_weekly_return_per_market(paper_dir):
    df = load_paper_weekly_trend(str(paper_dir))
    jp = df[df["market"] == "jp"][CUM_WEEKLY_COLUMN].tolist()
    assert jp[0] == pytest.approx(0.02)
    assert jp[1] == pytest.approx(1.02 * 0.99 - 1)
    assert jp[2] == pytest.approx(1.02 * 0.99 * 1.03 - 1)
    assert jp[3] == pytest.approx(1.02 * 0.99 * 1.03 * 1.01 - 1)
    us = df[df["market"] == "us"][CUM_WEEKLY_COLUMN].tolist()
    assert us[0] == pytest.approx(0.0)
    assert math.isnan(us[1])


def test_cumulative_excess_is_split_by_basis(paper_dir):
    df = load_paper_weekly_trend(str(paper_dir))
    jp = df[df["market"] == "jp"][CUM_EXCESS_COLUMN].tolist()
    # 印なしの古い週どうし
    assert jp[0] == pytest.approx(0.01)
    assert jp[1] == pytest.approx(1.01 * 0.99 - 1)
    # exposure_scaled の週は古い週を掛けずに始め直す
    assert jp[2] == pytest.approx(0.02)
    assert jp[3] == pytest.approx(1.02 * 1.01 - 1)
    us = df[df["market"] == "us"][CUM_EXCESS_COLUMN].tolist()
    assert us[0] == pytest.approx(-0.005)
    assert math.isnan(us[1])


def test_broken_files_are_skipped(paper_dir):
    reports = load_paper_weekly_reports(str(paper_dir))
    assert [r["week"] for r in reports] == ["2026-W43", "2026-W44", "2026-W45", "2026-W46"]
    df = load_paper_weekly_trend(str(paper_dir))
    assert "2026-W42" not in set(df["week"])
    assert "2026-W41" not in set(df["week"])


def test_report_without_markets_is_skipped():
    df = build_paper_weekly_trend([{"week": "2026-W40"}, {"week": "2026-W41", "markets": []}])
    assert df.empty
    assert CUM_WEEKLY_COLUMN in df.columns
    assert CUM_EXCESS_COLUMN in df.columns


def test_empty_dir_gives_no_data(tmp_path):
    df = load_paper_weekly_trend(str(tmp_path))
    assert df.empty
    assert format_paper_weekly_markdown(df) == "データなし"


def test_count_columns_for_fill_price_and_missing_close(tmp_path):
    """約定の値段で評価した銘柄・週末の終値の無い銘柄の数を右端の2列に出す（#2062）。"""
    filled = _new_market(0.01, 0.0, 0.01, 0.01, 0.4, partial=True)
    filled["valued_at_fill_price"] = ["7203", "9984"]
    filled["missing_week_end_close"] = ["6758"]
    legacy = _old_market(0.02, 0.01, 0.01)
    del legacy["valued_at_fill_price"]
    del legacy["missing_week_end_close"]
    _write(tmp_path / "2026-W44.json", "2026-W44", "2026-10-26", "2026-10-30", {"jp": legacy})
    _write(
        tmp_path / "2026-W45.json",
        "2026-W45",
        "2026-11-02",
        "2026-11-06",
        {"jp": _new_market(0.03, 0.02, 0.02, 0.01, 0.5)},
    )
    _write(tmp_path / "2026-W46.json", "2026-W46", "2026-11-09", "2026-11-13", {"jp": filled})

    df = load_paper_weekly_trend(str(tmp_path))
    assert list(df.columns[-2:]) == [
        "valued_at_fill_price_count",
        "missing_week_end_close_count",
    ]
    assert list(df.columns[-4:-2]) == [CUM_WEEKLY_COLUMN, CUM_EXCESS_COLUMN]
    legacy_row, empty_row, filled_row = (df.iloc[i] for i in range(3))
    assert math.isnan(legacy_row["valued_at_fill_price_count"])
    assert math.isnan(legacy_row["missing_week_end_close_count"])
    assert empty_row["valued_at_fill_price_count"] == 0
    assert empty_row["missing_week_end_close_count"] == 0
    assert filled_row["valued_at_fill_price_count"] == 2
    assert filled_row["missing_week_end_close_count"] == 1

    markdown = format_paper_weekly_markdown(df)
    columns = [c.strip() for c in markdown.splitlines()[0].strip("|").split("|")]
    legacy_line = [line for line in markdown.splitlines() if "2026-W44" in line][0]
    cells = [c.strip() for c in legacy_line.strip("|").split("|")]
    assert cells[columns.index("valued_at_fill_price_count")] == ""
    assert cells[columns.index("missing_week_end_close_count")] == ""
    filled_line = [line for line in markdown.splitlines() if "2026-W46" in line][0]
    cells = [c.strip() for c in filled_line.strip("|").split("|")]
    assert cells[columns.index("valued_at_fill_price_count")] == "2.000000"
