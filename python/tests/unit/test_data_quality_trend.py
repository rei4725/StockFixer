"""data_quality_log の流れの表（#1951）のテスト。期待値は手で数えた値。"""

from contextlib import contextmanager
from datetime import datetime, timezone

import src.utils.db.quality_log as quality_log
from src.reporting.data_quality_trend import (
    build_data_quality_trend,
    format_data_quality_trend_markdown,
)
from src.utils.db.quality_log import QualityLogRecord, load_quality_logs


def _rec(market, symbol, check, level, checked_at):
    return QualityLogRecord(
        market=market,
        symbol=symbol,
        check_name=check,
        level=level,
        detail="d",
        checked_at=checked_at,
    )


def test_build_trend_counts_by_day_market_check_and_level():
    records = [
        # 11-10 jp nan_rate warning: AAPL 2回（同じ日）と MSFT 1回 → 件数3、銘柄2
        _rec("jp", "7203", "nan_rate", "warning", "2026-11-10T01:00:00+00:00"),
        _rec("jp", "7203", "nan_rate", "warning", "2026-11-10T09:00:00+00:00"),
        _rec("jp", "6758", "nan_rate", "warning", "2026-11-10T01:00:00+00:00"),
        # 11-10 jp nan_rate error は別の行
        _rec("jp", "7203", "nan_rate", "error", "2026-11-10T01:00:00+00:00"),
        # 11-10 us price_step warning 1件
        _rec("us", "AAPL", "price_step", "warning", "2026-11-10T22:00:00+00:00"),
        # 11-11 jp consecutive_missing warning 1件（日が違うので別の行）
        _rec("jp", "7203", "consecutive_missing", "warning", "2026-11-11T01:00:00+00:00"),
        # 知らない check_name もそのまま行にする
        _rec("jp", "7203", "brand_new_check", "warning", "2026-11-11T02:00:00+00:00"),
    ]
    df = build_data_quality_trend(records)

    assert list(df.columns) == ["date", "market", "check_name", "level", "count", "symbols"]
    assert df.values.tolist() == [
        ["2026-11-10", "jp", "nan_rate", "error", 1, 1],
        ["2026-11-10", "jp", "nan_rate", "warning", 3, 2],
        ["2026-11-10", "us", "price_step", "warning", 1, 1],
        ["2026-11-11", "jp", "brand_new_check", "warning", 1, 1],
        ["2026-11-11", "jp", "consecutive_missing", "warning", 1, 1],
    ]

    md = format_data_quality_trend_markdown(df)
    lines = md.splitlines()
    assert lines[0] == "| date | market | check_name | level | count | symbols |"
    assert lines[3] == "| 2026-11-10 | jp | nan_rate | warning | 3 | 2 |"


def test_same_symbol_on_different_days_counts_each_day():
    records = [
        _rec("jp", "7203", "nan_rate", "warning", "2026-11-10T23:59:59+00:00"),
        _rec("jp", "7203", "nan_rate", "warning", "2026-11-11T00:00:00+00:00"),
    ]
    df = build_data_quality_trend(records)
    assert df.values.tolist() == [
        ["2026-11-10", "jp", "nan_rate", "warning", 1, 1],
        ["2026-11-11", "jp", "nan_rate", "warning", 1, 1],
    ]


def test_empty_input_gives_empty_table():
    df = build_data_quality_trend([])
    assert df.empty
    assert list(df.columns) == ["date", "market", "check_name", "level", "count", "symbols"]
    assert format_data_quality_trend_markdown(df) == "データなし"


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, log, rows):
        self._log = log
        self._rows = rows

    def execute(self, sql, params=None):
        self._log.append((sql.strip(), params))
        return _FakeCursor(self._rows)


def _patch_connection(monkeypatch, rows):
    log: list = []

    @contextmanager
    def fake_connection():
        yield _FakeConnection(log, rows)

    monkeypatch.setattr(quality_log, "_db_connection", fake_connection)
    return log


def test_load_quality_logs_only_selects_in_range(monkeypatch):
    rows = [("jp", "7203", "nan_rate", "warning", "nan 0.3", "2026-11-10T01:00:00+00:00")]
    log = _patch_connection(monkeypatch, rows)

    records = load_quality_logs(
        since=datetime(2026, 11, 1),
        until=datetime(2026, 11, 15, tzinfo=timezone.utc),
        market="jp",
    )

    assert len(log) == 1
    sql, params = log[0]
    assert sql.upper().startswith("SELECT")
    for word in ("INSERT", "UPDATE", "DELETE"):
        assert word not in sql.upper()
    # 日時は checked_at と同じ ISO 8601 の UTC の文字列で比べる（tz の無い値は UTC とみなす）
    assert params == ["2026-11-01T00:00:00+00:00", "2026-11-15T00:00:00+00:00", "jp"]
    assert records == [
        QualityLogRecord(
            market="jp",
            symbol="7203",
            check_name="nan_rate",
            level="warning",
            detail="nan 0.3",
            checked_at="2026-11-10T01:00:00+00:00",
        )
    ]


def test_load_quality_logs_without_filters(monkeypatch):
    log = _patch_connection(monkeypatch, [])
    assert load_quality_logs() == []
    assert "WHERE" not in log[-1][0].upper()
    assert log[-1][1] == []
