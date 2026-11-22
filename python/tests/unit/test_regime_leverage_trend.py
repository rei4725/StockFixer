"""regime_leverage_log の日ごとの流れの表（#2094）のテスト。DB には触らない。"""

import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import pytest
import run_regime_leverage_trend

from src.reporting.regime_leverage_trend import (
    build_regime_leverage_trend,
    format_regime_leverage_trend_markdown,
    to_jst_date,
)
from src.trading.regime_leverage_strategy.types import RegimeLeverageSnapshot

_COLUMNS = [
    "date",
    "rows",
    "action",
    "reason",
    "reasons",
    "shares",
    "equity_now_jpy",
    "equity_at_entry_jpy",
    "change_from_entry",
    "maintenance_ratio",
    "stop_price_jpy",
]


def _row(
    row_id: int,
    executed_at: datetime,
    action: str,
    reason: str,
    shares: float,
    equity_now: float,
    equity_at_entry: Optional[float],
    maintenance_ratio: Optional[float] = None,
    stop_price: Optional[float] = None,
) -> RegimeLeverageSnapshot:
    return RegimeLeverageSnapshot(
        id=row_id,
        executed_at=executed_at,
        action=action,
        reason=reason,
        spy_price_usd=600.0,
        usdjpy_rate=150.0,
        shares=shares,
        entry_date=None,
        entry_price_jpy=None,
        entry_commission_jpy=None,
        equity_at_entry_jpy=equity_at_entry,
        stop_price_jpy=stop_price,
        equity_now_jpy=equity_now,
        maintenance_ratio=maintenance_ratio,
    )


def _sample_rows() -> list[RegimeLeverageSnapshot]:
    return [
        # 日本時間 11-17: 建玉の無い日（equity_at_entry_jpy が None）
        _row(1, datetime(2026, 11, 16, 22, 30), "noop", "weekly_noop", 0.0, 1_000_000.0, None),
        # 日本時間 11-18: 同じ日に daily_noop と regime_entry が並ぶ（id の順で最後は entry）
        _row(3, datetime(2026, 11, 17, 23, 0), "entry", "regime_entry", 16.0, 1_000_000.0,
             1_000_000.0, 0.55, 115_000.0),
        _row(2, datetime(2026, 11, 17, 22, 30), "noop", "daily_noop", 0.0, 1_000_000.0, None),
        # 日本時間 11-19: 建玉ありで評価額が 2% 増える
        _row(4, datetime(2026, 11, 18, 22, 30), "noop", "daily_noop", 16.0, 1_020_000.0,
             1_000_000.0, 0.52, 115_000.0),
    ]  # fmt: skip


def test_to_jst_date_treats_naive_time_as_utc():
    assert to_jst_date(datetime(2026, 11, 17, 22, 30)).isoformat() == "2026-11-18"
    assert to_jst_date(datetime(2026, 11, 17, 14, 59)).isoformat() == "2026-11-17"


def test_build_trend_one_row_per_jst_day_with_last_row_state():
    df = build_regime_leverage_trend(_sample_rows())

    assert list(df.columns) == _COLUMNS
    assert df["date"].tolist() == ["2026-11-17", "2026-11-18", "2026-11-19"]
    assert df["rows"].tolist() == [1, 2, 1]

    mixed = df.iloc[1]
    assert mixed["action"] == "entry"
    assert mixed["reason"] == "regime_entry"
    assert mixed["reasons"] == "daily_noop; regime_entry"
    assert mixed["shares"] == 16.0
    assert mixed["change_from_entry"] == pytest.approx(0.0)
    assert mixed["maintenance_ratio"] == pytest.approx(0.55)
    assert mixed["stop_price_jpy"] == pytest.approx(115_000.0)

    last = df.iloc[2]
    assert last["change_from_entry"] == pytest.approx(0.02)
    assert last["equity_now_jpy"] == pytest.approx(1_020_000.0)


def test_build_trend_day_without_position_leaves_change_empty():
    df = build_regime_leverage_trend(_sample_rows())

    flat = df.iloc[0]
    assert flat["action"] == "noop"
    assert flat["reason"] == "weekly_noop"
    assert flat["shares"] == 0.0
    for column in ("equity_at_entry_jpy", "change_from_entry", "maintenance_ratio"):
        assert pd.isna(flat[column])
    assert pd.isna(flat["stop_price_jpy"])
    markdown_lines = format_regime_leverage_trend_markdown(df).splitlines()
    assert markdown_lines[2] == (
        "| 2026-11-17 | 1 | noop | weekly_noop | weekly_noop | 0.0000 | 1000000.0000 |  |  |  |  |"
    )
    csv_lines = df.to_csv(index=False).splitlines()
    assert csv_lines[1] == "2026-11-17,1,noop,weekly_noop,weekly_noop,0.0,1000000.0,,,,"


def test_build_trend_with_no_rows_is_empty_with_columns():
    df = build_regime_leverage_trend([])

    assert df.empty
    assert list(df.columns) == _COLUMNS
    assert format_regime_leverage_trend_markdown(df) == "データなし"
    assert df.to_csv(index=False).strip() == ",".join(_COLUMNS)


def test_cli_writes_file_only_with_output(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_load(since=None):
        calls.append(since)
        return _sample_rows()

    monkeypatch.setattr(run_regime_leverage_trend, "load_regime_leverage_log", fake_load)

    monkeypatch.setattr(sys, "argv", ["run_regime_leverage_trend.py", "--format", "csv"])
    run_regime_leverage_trend.main()
    out = capsys.readouterr().out
    assert out.splitlines()[0] == ",".join(_COLUMNS)
    assert list(tmp_path.iterdir()) == []

    target = tmp_path / "trend.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_regime_leverage_trend.py", "--since", "2026-11-01", "--output", str(target)],
    )
    run_regime_leverage_trend.main()
    assert capsys.readouterr().out == ""
    assert target.read_text(encoding="utf-8").startswith("| date | rows | action |")
    assert calls == [None, datetime(2026, 11, 1)]
