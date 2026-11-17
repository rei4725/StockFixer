"""results/orders と results/orders/rule の発注の記録を読む表のテスト（#1836）。"""

import json
import logging

import pandas as pd
import pytest

from src.reporting.order_gate_trend import (
    ROUTE_MORNING,
    ROUTE_RULE,
    build_order_gate_trend,
    format_order_gate_csv,
    format_order_gate_markdown,
    load_order_gate_records,
    load_order_gate_trend,
)


def _morning(day, **overrides):
    record = {
        "date": day,
        "written_at": f"{day}T08:50:00+09:00",
        "market": "jp",
        "mode": "paper",
        "entry_error": None,
        "token_error": None,
        "trading_stopped": False,
        "correlation_blocked": False,
        "buy_orders": 0,
        "sell_orders": 1,
        "short_orders": 0,
        "skipped": 240,
        "errors": 0,
        "skip_reasons": {"sector_cap": 211, "max_orders_per_run": 24},
        "sl_tp_unevaluable_symbols": [],
        "sl_tp_unevaluable_count": 0,
        "held_without_prediction_symbols": [],
        "held_without_prediction_count": 0,
    }
    record.update(overrides)
    return record


def _rule(day, **overrides):
    record = {
        "date": day,
        "written_at": f"{day}T16:30:00+09:00",
        "market": "jp",
        "allow_new_buys": True,
        "new_buy_step_error": None,
        "skip_reasons": {"no_signal": 140, "sell_not_held": 54},
        "buy_orders": 0,
        "sell_orders": 0,
        "skipped": 194,
        "errors": 0,
        "unavailable_count": 0,
        "unavailable_held_count": 0,
        "unjudged_held_symbols": ["3350"],
        "unjudged_held_count": 1,
    }
    record.update(overrides)
    return record


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def orders_dir(tmp_path):
    _write(tmp_path / "2026-11-12.json", _morning("2026-11-12"))
    _write(tmp_path / "rule" / "2026-11-12.json", _rule("2026-11-12"))
    # 11-13: 8:50 は理由の鍵が違い、保有の判定をしなかった日（null）
    _write(
        tmp_path / "2026-11-13.json",
        _morning(
            "2026-11-13",
            trading_stopped=True,
            skip_reasons={"daily_loss": 5},
            held_without_prediction_symbols=None,
            held_without_prediction_count=None,
            token_error="RuntimeError('x')",
        ),
    )
    # 11-13: 16:30 は保有を読めなかった日（null）
    _write(
        tmp_path / "rule" / "2026-11-13.json",
        _rule(
            "2026-11-13",
            allow_new_buys=False,
            new_buy_step_error="ValueError('y')",
            unjudged_held_symbols=None,
            unjudged_held_count=None,
            unavailable_held_count=None,
        ),
    )
    # 11-14: 16:30 の道しか無い日
    _write(tmp_path / "rule" / "2026-11-14.json", _rule("2026-11-14", buy_orders=2))
    return tmp_path


def _at(df, day, route):
    rows = df[(df["date"] == day) & (df["route"] == route)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_rows_per_day_and_route(orders_dir):
    df = load_order_gate_trend(str(orders_dir))
    assert list(zip(df["date"], df["route"])) == [
        ("2026-11-12", ROUTE_MORNING),
        ("2026-11-12", ROUTE_RULE),
        ("2026-11-13", ROUTE_MORNING),
        ("2026-11-13", ROUTE_RULE),
        ("2026-11-14", ROUTE_RULE),
    ]
    assert _at(df, "2026-11-14", ROUTE_RULE)["buy_orders"] == 2
    assert _at(df, "2026-11-12", ROUTE_MORNING)["mode"] == "paper"
    assert pd.isna(_at(df, "2026-11-12", ROUTE_RULE)["mode"])


def test_skip_reasons_columns_union_with_zero_for_absent_key(orders_dir):
    df = load_order_gate_trend(str(orders_dir))
    for key in ("daily_loss", "max_orders_per_run", "no_signal", "sector_cap", "sell_not_held"):
        assert f"skip_{key}" in df.columns
    morning12 = _at(df, "2026-11-12", ROUTE_MORNING)
    assert morning12["skip_sector_cap"] == 211
    assert morning12["skip_max_orders_per_run"] == 24
    assert morning12["skip_daily_loss"] == 0
    assert morning12["skip_no_signal"] == 0
    morning13 = _at(df, "2026-11-13", ROUTE_MORNING)
    assert morning13["skip_daily_loss"] == 5
    assert morning13["skip_sector_cap"] == 0
    assert _at(df, "2026-11-13", ROUTE_RULE)["skip_no_signal"] == 140


def test_null_counts_stay_apart_from_zero(orders_dir):
    df = load_order_gate_trend(str(orders_dir))
    assert _at(df, "2026-11-12", ROUTE_MORNING)["held_without_prediction_count"] == 0
    assert pd.isna(_at(df, "2026-11-13", ROUTE_MORNING)["held_without_prediction_count"])
    assert _at(df, "2026-11-12", ROUTE_RULE)["unjudged_held_count"] == 1
    assert pd.isna(_at(df, "2026-11-13", ROUTE_RULE)["unjudged_held_count"])
    assert pd.isna(_at(df, "2026-11-13", ROUTE_RULE)["unavailable_held_count"])
    # 道に無い数（16:30 の short_orders）も欠けにする
    assert pd.isna(_at(df, "2026-11-12", ROUTE_RULE)["short_orders"])

    csv_lines = format_order_gate_csv(df).splitlines()
    header = csv_lines[0].split(",")
    column = header.index("held_without_prediction_count")
    assert csv_lines[1].split(",")[column] == "0"
    assert csv_lines[3].split(",")[column] == "null"


def test_flags_and_error_presence(orders_dir):
    df = load_order_gate_trend(str(orders_dir))
    morning12 = _at(df, "2026-11-12", ROUTE_MORNING)
    assert not morning12["trading_stopped"]
    assert not morning12["token_error"]
    assert not morning12["entry_error"]
    assert pd.isna(morning12["allow_new_buys"])
    assert pd.isna(morning12["new_buy_step_error"])
    morning13 = _at(df, "2026-11-13", ROUTE_MORNING)
    assert morning13["trading_stopped"]
    assert morning13["token_error"]
    rule13 = _at(df, "2026-11-13", ROUTE_RULE)
    assert not rule13["allow_new_buys"]
    assert rule13["new_buy_step_error"]
    assert pd.isna(rule13["trading_stopped"])


def test_unreadable_file_warns_and_is_skipped(orders_dir, caplog):
    (orders_dir / "2026-11-11.json").write_text("{not json", encoding="utf-8")
    (orders_dir / "rule" / "2026-11-11.json").write_text("[1, 2]", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        records = load_order_gate_records(str(orders_dir))
    assert len(records) == 5
    assert "2026-11-11" not in {r["date"] for r in records}
    assert sum(r.levelno == logging.WARNING for r in caplog.records) >= 2
    assert len(build_order_gate_trend(records)) == 5


def test_missing_skip_reasons_is_null(tmp_path):
    record = _morning("2026-11-10")
    del record["skip_reasons"]
    _write(tmp_path / "2026-11-10.json", record)
    _write(tmp_path / "2026-11-11.json", _morning("2026-11-11"))
    df = load_order_gate_trend(str(tmp_path))
    assert pd.isna(_at(df, "2026-11-10", ROUTE_MORNING)["skip_sector_cap"])
    assert _at(df, "2026-11-11", ROUTE_MORNING)["skip_sector_cap"] == 211


def test_empty_dir_gives_no_rows(tmp_path):
    df = load_order_gate_trend(str(tmp_path))
    assert df.empty
    assert format_order_gate_markdown(df) == "データなし"


def test_markdown_writes_null(orders_dir):
    text = format_order_gate_markdown(load_order_gate_trend(str(orders_dir)))
    lines = text.splitlines()
    assert lines[0].startswith("| date | route |")
    assert len(lines) == 2 + 5
    assert "null" in lines[4]


def test_does_not_write_files(orders_dir):
    before = sorted(p.relative_to(orders_dir) for p in orders_dir.rglob("*"))
    load_order_gate_trend(str(orders_dir))
    after = sorted(p.relative_to(orders_dir) for p in orders_dir.rglob("*"))
    assert before == after


# 足す前の表の列（#1958 の前）。足した列はこの後ろに並ぶ
_BEFORE_COLUMNS = [
    "date",
    "route",
    "market",
    "mode",
    "buy_orders",
    "sell_orders",
    "short_orders",
    "skipped",
    "errors",
    "trading_stopped",
    "correlation_blocked",
    "allow_new_buys",
    "entry_error",
    "token_error",
    "new_buy_step_error",
    "held_without_prediction_count",
    "sl_tp_unevaluable_count",
    "unjudged_held_count",
    "unavailable_held_count",
    "skip_daily_loss",
    "skip_max_orders_per_run",
    "skip_no_signal",
    "skip_sector_cap",
    "skip_sell_not_held",
]
_GATE_COLUMNS = [
    "enc",
    "avg_correlation",
    "daily_loss",
    "daily_loss_limit",
    "threshold_scale",
    "skipped_min_change",
    "unavailable_count",
    "risk_gate_allowed",
    "correlation_gate_allowed",
    "reason_code",
    "stale_prediction_reason",
]
_MORNING_GATE = {
    "reason_code": "daily_loss_limit",
    "daily_loss": 10800.0,
    "daily_loss_limit": 195448.8,
    "enc": 2.225,
    "avg_correlation": 0.266,
    "stale_prediction_reason": None,
    "threshold_scale": 0.5,
    "skipped_min_change": 3,
}
_RULE_GATE = {
    "risk_gate_allowed": True,
    "risk_gate_reason_code": "ok",
    "correlation_gate_allowed": False,
    "enc": 2.242,
    "unavailable_count": 4,
}


def _gate_records():
    return [
        _morning("2026-11-12", route=ROUTE_MORNING, **_MORNING_GATE),
        _rule("2026-11-12", route=ROUTE_RULE, **_RULE_GATE),
        # 11-13: 8:50 は値が null の記録、16:30 はゲートの鍵の無い記録
        _morning(
            "2026-11-13",
            route=ROUTE_MORNING,
            skip_reasons={"daily_loss": 5},
            reason_code=None,
            daily_loss=None,
            daily_loss_limit=None,
            enc=None,
            avg_correlation=None,
            stale_prediction_reason="stale",
            threshold_scale=None,
            skipped_min_change=None,
        ),
        _rule("2026-11-13", route=ROUTE_RULE, unavailable_count=None),
    ]


def _without_gate_keys(record):
    gate_keys = set(_MORNING_GATE) | set(_RULE_GATE)
    return {k: v for k, v in record.items() if k not in gate_keys}


def test_before_columns_keep_order_and_values():
    records = _gate_records()
    df = build_order_gate_trend(records)
    assert list(df.columns) == _BEFORE_COLUMNS + _GATE_COLUMNS
    before = build_order_gate_trend([_without_gate_keys(r) for r in records])
    pd.testing.assert_frame_equal(df[_BEFORE_COLUMNS], before[_BEFORE_COLUMNS])
    assert list(zip(df["date"], df["route"])) == [
        ("2026-11-12", ROUTE_MORNING),
        ("2026-11-12", ROUTE_RULE),
        ("2026-11-13", ROUTE_MORNING),
        ("2026-11-13", ROUTE_RULE),
    ]


def test_gate_columns_by_route():
    df = build_order_gate_trend(_gate_records())
    morning = _at(df, "2026-11-12", ROUTE_MORNING)
    assert morning["enc"] == pytest.approx(2.225)
    assert morning["avg_correlation"] == pytest.approx(0.266)
    assert morning["daily_loss"] == pytest.approx(10800.0)
    assert morning["daily_loss_limit"] == pytest.approx(195448.8)
    assert morning["threshold_scale"] == pytest.approx(0.5)
    assert morning["skipped_min_change"] == 3
    assert morning["reason_code"] == "daily_loss_limit"
    assert not morning["stale_prediction_reason"]
    # 16:30 だけにある鍵は 8:50 の行で欠け
    for key in ("unavailable_count", "risk_gate_allowed", "correlation_gate_allowed"):
        assert pd.isna(morning[key])

    rule = _at(df, "2026-11-12", ROUTE_RULE)
    assert rule["enc"] == pytest.approx(2.242)
    assert rule["unavailable_count"] == 4
    assert rule["risk_gate_allowed"]
    assert not rule["correlation_gate_allowed"]
    assert rule["reason_code"] == "ok"
    # 8:50 だけにある鍵は 16:30 の行で欠け
    for key in (
        "avg_correlation",
        "daily_loss",
        "daily_loss_limit",
        "threshold_scale",
        "skipped_min_change",
        "stale_prediction_reason",
    ):
        assert pd.isna(rule[key])

    assert str(df["enc"].dtype) == "Float64"
    assert str(df["skipped_min_change"].dtype) == "Int64"
    assert str(df["risk_gate_allowed"].dtype) == "boolean"


def test_gate_null_and_missing_stay_apart_from_zero():
    records = _gate_records()
    records.append(
        _morning("2026-11-14", route=ROUTE_MORNING, daily_loss=0.0, skipped_min_change=0, enc=0.0)
    )
    df = build_order_gate_trend(records)
    null_morning = _at(df, "2026-11-13", ROUTE_MORNING)
    for key in (
        "enc",
        "avg_correlation",
        "daily_loss",
        "daily_loss_limit",
        "threshold_scale",
        "skipped_min_change",
        "reason_code",
    ):
        assert pd.isna(null_morning[key])
    assert null_morning["stale_prediction_reason"]
    missing_rule = _at(df, "2026-11-13", ROUTE_RULE)
    for key in (
        "enc",
        "unavailable_count",
        "risk_gate_allowed",
        "correlation_gate_allowed",
        "reason_code",
    ):
        assert pd.isna(missing_rule[key])
    zero_morning = _at(df, "2026-11-14", ROUTE_MORNING)
    assert zero_morning["daily_loss"] == 0
    assert zero_morning["skipped_min_change"] == 0
    assert zero_morning["enc"] == 0
    # 鍵の無い記録の stale_prediction_reason は欠け
    assert pd.isna(zero_morning["stale_prediction_reason"])

    csv_lines = format_order_gate_csv(df).splitlines()
    column = csv_lines[0].split(",").index("daily_loss")
    assert csv_lines[3].split(",")[column] == "null"
    assert csv_lines[5].split(",")[column] == "0.0"
