"""8:50 の発注の記録の sector_cap_by_sector を読み返す表のテスト（#2043）。"""

import json
import logging

import pandas as pd

from src.reporting.order_sector_cap_trend import (
    SectorCapDay,
    build_sector_cap_daily,
    build_sector_cap_summary,
    format_sector_cap_csv,
    format_sector_cap_markdown,
    load_sector_cap_days,
    tail_days,
    to_sector_cap_days,
)


def _morning(day, by_sector=None, **overrides):
    record = {
        "date": day,
        "market": "jp",
        "mode": "paper",
        "skipped": 235,
        "skip_reasons": {"sector_cap": sum((by_sector or {}).values()), "max_orders_per_run": 25},
    }
    if by_sector is not None:
        record["sector_cap_by_sector"] = by_sector
    record.update(overrides)
    return record


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_four_shapes_of_records(tmp_path, caplog):
    # 鍵のある記録
    _write(
        tmp_path / "2026-11-17.json",
        _morning("2026-11-17", {"Industrials": 62, "Technology": 30}),
    )
    # 鍵の無い古い記録
    old = _morning("2026-11-10")
    _write(tmp_path / "2026-11-10.json", old)
    # 空の {}
    _write(tmp_path / "2026-11-18.json", _morning("2026-11-18", {}))
    # 読めないファイル
    (tmp_path / "2026-11-19.json").write_text("{not json", encoding="utf-8")
    # 16:30 の記録は読まない
    (tmp_path / "rule").mkdir()
    _write(tmp_path / "rule" / "2026-11-17.json", _morning("2026-11-17", {"Energy": 5}))

    with caplog.at_level(logging.WARNING):
        days = load_sector_cap_days(str(tmp_path))

    assert "sector_cap_by_sector" not in old
    assert [d.date for d in days] == ["2026-11-17", "2026-11-18"]
    assert days[1].by_sector == {}
    assert any("2026-11-19.json" in r.getMessage() for r in caplog.records)

    daily = build_sector_cap_daily(days)
    assert list(daily.columns) == ["date", "sector", "dropped"]
    assert daily.to_dict("records") == [
        {"date": "2026-11-17", "sector": "Industrials", "dropped": 62},
        {"date": "2026-11-17", "sector": "Technology", "dropped": 30},
    ]
    assert "Energy" not in set(daily["sector"])

    summary = build_sector_cap_summary(days)
    # 鍵の無い 11-10 は数えず、空の 11-18 は鍵のある日に数える
    assert summary["recorded_days"].tolist() == [2, 2]


def test_non_dict_shapes_are_skipped(tmp_path, caplog):
    _write(tmp_path / "2026-11-17.json", [1, 2, 3])
    _write(tmp_path / "2026-11-18.json", _morning("2026-11-18", None, sector_cap_by_sector=None))
    with caplog.at_level(logging.WARNING):
        days = load_sector_cap_days(str(tmp_path))
    assert days == []
    assert any("形が違う" in r.getMessage() for r in caplog.records)
    assert build_sector_cap_daily(days).empty
    assert build_sector_cap_summary(days).empty


def test_daily_sum_matches_skip_reasons_sector_cap():
    by_sector = {"Industrials": 63, "Consumer Cyclical": 34, "Technology": 28, "Healthcare": 85}
    record = _morning("2026-11-19", by_sector)
    assert record["skip_reasons"]["sector_cap"] == 210
    daily = build_sector_cap_daily(to_sector_cap_days([record]))
    day = daily[daily["date"] == "2026-11-19"]
    assert int(day["dropped"].sum()) == record["skip_reasons"]["sector_cap"]
    # 日の中は落とした数の多い順
    assert day["sector"].tolist() == [
        "Healthcare",
        "Industrials",
        "Consumer Cyclical",
        "Technology",
    ]


def test_non_numeric_values_are_dropped_and_date_from_filename(tmp_path):
    record = {"sector_cap_by_sector": {"A": 3, "B": None, "C": "x", "D": True, "E": 0}}
    _write(tmp_path / "2026-11-20.json", record)
    days = load_sector_cap_days(str(tmp_path))
    assert days == [SectorCapDay(date="2026-11-20", by_sector={"A": 3, "E": 0})]


def test_summary_totals_and_days():
    days = [
        SectorCapDay("2026-11-17", {"Industrials": 62, "Technology": 28}),
        SectorCapDay("2026-11-18", {"Industrials": 63, "Energy": 0}),
        SectorCapDay("2026-11-19", {}),
    ]
    summary = build_sector_cap_summary(days)
    assert summary.to_dict("records") == [
        {"sector": "Industrials", "dropped": 125, "days": 2, "recorded_days": 3},
        {"sector": "Technology", "dropped": 28, "days": 1, "recorded_days": 3},
        {"sector": "Energy", "dropped": 0, "days": 0, "recorded_days": 3},
    ]
    assert str(summary["dropped"].dtype) == "Int64"


def test_daily_sorted_by_date_even_if_input_unsorted():
    records = [_morning("2026-11-18", {"B": 1}), _morning("2026-11-17", {"A": 2})]
    daily = build_sector_cap_daily(to_sector_cap_days(records))
    assert daily["date"].tolist() == ["2026-11-17", "2026-11-18"]


def test_tail_days():
    days = [SectorCapDay(f"2026-11-1{i}", {}) for i in range(5)]
    assert [d.date for d in tail_days(days, 2)] == ["2026-11-13", "2026-11-14"]
    assert tail_days(days, None) == days
    assert tail_days(days, 0) == days


def test_formats():
    daily = build_sector_cap_daily([SectorCapDay("2026-11-17", {"Industrials": 62})])
    md = format_sector_cap_markdown(daily)
    assert md.splitlines() == [
        "| date | sector | dropped |",
        "| --- | --- | --- |",
        "| 2026-11-17 | Industrials | 62 |",
    ]
    assert format_sector_cap_csv(daily).splitlines() == [
        "date,sector,dropped",
        "2026-11-17,Industrials,62",
    ]
    assert format_sector_cap_markdown(pd.DataFrame()) == "データなし"
