"""16:30 のルールの道の記録の zero_qty_symbols を読み返す表のテスト（#2189）。"""

import json
import logging
import sys

import pandas as pd
import run_rule_zero_qty_trend

from src.reporting.rule_zero_qty_trend import (
    ZeroQtyDay,
    ZeroQtyRow,
    build_zero_qty_daily,
    build_zero_qty_summary,
    format_zero_qty_csv,
    format_zero_qty_markdown,
    load_zero_qty_days,
    tail_days,
    to_zero_qty_days,
)

BUDGET = 100000.0


def _zero(symbol, reason="zero_qty_budget_cap", price=2000.0, lot=100, sized_qty=100):
    return {
        "symbol": symbol,
        "reason": reason,
        "price": price,
        "lot_cost": price * lot,
        "sized_qty": sized_qty,
        "budget": BUDGET,
    }


def _rule(day, zero_rows=None, **overrides):
    record = {
        "date": day,
        "market": "jp",
        "allow_new_buys": True,
        "skip_reasons": {"zero_qty_budget_cap": len(zero_rows or [])},
        "buy_orders": 0,
    }
    if zero_rows is not None:
        record["zero_qty_symbols"] = zero_rows
        record["zero_qty_count"] = len(zero_rows)
    record.update(overrides)
    return record


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_same_symbol_on_two_days_counts_two(tmp_path):
    _write(tmp_path / "2026-11-18.json", _rule("2026-11-18", [_zero("9766"), _zero("3099")]))
    _write(
        tmp_path / "2026-11-19.json",
        _rule("2026-11-19", [_zero("9766", price=2500.0), _zero("7203")]),
    )

    summary = build_zero_qty_summary(load_zero_qty_days(str(tmp_path)))

    by_symbol = summary.set_index("symbol")
    assert by_symbol.loc["9766", "days"] == 2
    assert by_symbol.loc["9766", "recorded_days"] == 2
    assert by_symbol.loc["9766", "last_date"] == "2026-11-19"
    assert by_symbol.loc["9766", "last_lot_cost"] == 250000.0
    assert by_symbol.loc["9766", "last_budget"] == BUDGET
    assert by_symbol.loc["9766", "reason_days"] == "zero_qty_budget_cap=2"
    assert by_symbol.loc["3099", "days"] == 1
    assert by_symbol.loc["7203", "days"] == 1
    assert list(summary["symbol"]) == ["9766", "3099", "7203"]


def test_record_without_key_makes_no_row_and_no_recorded_day(tmp_path):
    old = _rule("2026-11-17", skip_reasons={"zero_qty_budget_cap": 7})
    _write(tmp_path / "2026-11-17.json", old)
    _write(tmp_path / "2026-11-18.json", _rule("2026-11-18", [_zero("9766")]))

    days = load_zero_qty_days(str(tmp_path))

    assert "zero_qty_symbols" not in old
    assert [d.date for d in days] == ["2026-11-18"]
    assert list(build_zero_qty_daily(days)["date"]) == ["2026-11-18"]
    assert build_zero_qty_summary(days).loc[0, "recorded_days"] == 1


def test_empty_list_day_counts_as_recorded(tmp_path):
    _write(tmp_path / "2026-11-19.json", _rule("2026-11-19", [_zero("9766")]))
    _write(tmp_path / "2026-11-20.json", _rule("2026-11-20", []))

    days = load_zero_qty_days(str(tmp_path))
    summary = build_zero_qty_summary(days)

    assert [d.date for d in days] == ["2026-11-19", "2026-11-20"]
    assert days[1].rows == ()
    assert summary.loc[0, "days"] == 1
    assert summary.loc[0, "recorded_days"] == 2
    assert len(build_zero_qty_daily(days)) == 1


def test_broken_json_is_skipped_with_warning(tmp_path, caplog):
    (tmp_path / "2026-11-18.json").write_text("{not json", encoding="utf-8")
    _write(tmp_path / "2026-11-19.json", ["not", "a", "dict"])
    _write(tmp_path / "2026-11-20.json", _rule("2026-11-20", [_zero("9766")]))

    with caplog.at_level(logging.WARNING):
        days = load_zero_qty_days(str(tmp_path))

    assert [d.date for d in days] == ["2026-11-20"]
    assert "読めず飛ばす" in caplog.text
    assert "形が違う" in caplog.text


def test_bad_rows_and_values():
    records = [
        _rule(
            "2026-11-19",
            [
                "not a dict",
                {"reason": "zero_qty_under_lot"},
                {"symbol": "5019", "reason": None, "price": "x", "lot_cost": True},
                _zero("5713", reason="zero_qty_under_lot", sized_qty=0),
            ],
        ),
        _rule("2026-11-18", [_zero("5713")]),
        {"date": "2026-11-17", "zero_qty_symbols": {"not": "a list"}},
    ]

    days = to_zero_qty_days(records)

    assert [d.date for d in days] == ["2026-11-18", "2026-11-19"]
    assert days[1].rows[0] == ZeroQtyRow(
        symbol="5019", reason="", price=None, lot_cost=None, budget=None, sized_qty=None
    )
    daily = build_zero_qty_daily(days)
    assert daily.loc[daily["symbol"] == "5019", "price"].isna().all()
    summary = build_zero_qty_summary(days).set_index("symbol")
    assert summary.loc["5713", "reason_days"] == "zero_qty_budget_cap=1;zero_qty_under_lot=1"
    assert summary.loc["5713", "last_reason"] == "zero_qty_under_lot"


def test_same_symbol_twice_in_one_day_counts_one_day():
    days = [
        ZeroQtyDay(
            date="2026-11-19",
            rows=tuple(ZeroQtyRow("9766", "r", 1.0, 100.0, 1.0, 0) for _ in range(2)),
        )
    ]

    assert build_zero_qty_summary(days).loc[0, "days"] == 1
    assert len(build_zero_qty_daily(days)) == 2


def test_tail_days():
    days = [ZeroQtyDay(date=f"2026-11-{d}", rows=()) for d in (18, 19, 20)]

    assert [d.date for d in tail_days(days, 2)] == ["2026-11-19", "2026-11-20"]
    assert tail_days(days, None) == days
    assert tail_days(days, 0) == days


def test_empty_tables_and_formats():
    assert format_zero_qty_markdown(build_zero_qty_summary([])) == "データなし"
    assert list(build_zero_qty_daily([]).columns) == [
        "date",
        "symbol",
        "reason",
        "price",
        "lot_cost",
        "budget",
        "sized_qty",
    ]
    df = pd.DataFrame({"a": [1, pd.NA]}, dtype="Int64")
    assert format_zero_qty_csv(df).splitlines() == ["a", "1", "null"]
    assert format_zero_qty_markdown(df) == "| a |\n| --- |\n| 1 |\n| null |"


def test_cli_summary_on_records_shaped_like_11_18_and_11_19(tmp_path, monkeypatch, capsys):
    seven = ["9766", "3099", "7911", "5019", "5713", "5803", "5801"]
    day18 = [_zero(s) for s in seven[:-1]] + [_zero(seven[-1], reason="zero_qty_under_lot")]
    day19 = [_zero(s) for s in seven] + [_zero(f"99{i:02d}") for i in range(4)]
    day19.append(_zero("9990", reason="zero_qty_under_lot"))
    _write(tmp_path / "2026-11-17.json", _rule("2026-11-17", skip_reasons={"zero_qty": 7}))
    _write(tmp_path / "2026-11-18.json", _rule("2026-11-18", day18))
    _write(tmp_path / "2026-11-19.json", _rule("2026-11-19", day19, buy_orders=1))
    for day in ("2026-11-20", "2026-11-24", "2026-11-25"):
        _write(tmp_path / f"{day}.json", _rule(day, []))
    monkeypatch.setattr(
        sys, "argv", ["run_rule_zero_qty_trend.py", "--dir", str(tmp_path), "--table", "summary"]
    )

    run_rule_zero_qty_trend.main()

    out = capsys.readouterr().out
    lines = out.splitlines()
    for symbol in seven:
        assert any(line.startswith(f"| {symbol} | 2 | 5 |") for line in lines), symbol
    assert sum(" | 1 | 5 |" in line for line in lines) == 5
    with capsys.disabled():
        print("\n" + out)
