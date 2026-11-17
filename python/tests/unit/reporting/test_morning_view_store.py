"""reporting.morning_view_store のテスト（#1875 段2）。

本番の DB には出ない。tmp の DuckDB に migration 0012 の SQL を流し、
Postgres の %s を DuckDB の ? に置き換えて保存・上書き・答え合わせ・読み出しを確かめる。
"""

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest

from src.reporting import morning_view_store as store
from src.reporting.morning_view import ViewRecord, tally

_MIGRATION = (
    Path(store.__file__).resolve().parents[1]
    / "utils"
    / "db"
    / "migrations"
    / "0012_add_morning_view_log_postgres.sql"
)


class _DuckDBConnection:
    def __init__(self, con):
        self._con = con

    def execute(self, sql, params=None):
        sql = sql.replace("%s", "?")
        if params is None:
            return self._con.execute(sql)
        return self._con.execute(sql, params)


@pytest.fixture
def con(tmp_path):
    raw = duckdb.connect(str(tmp_path / "mv.duckdb"))
    for statement in _MIGRATION.read_text(encoding="utf-8").split(";"):
        if statement.strip():
            raw.execute(statement)
    wrapped = _DuckDBConnection(raw)

    @contextmanager
    def _cm():
        yield wrapped

    with patch.object(store, "db_connection", _cm):
        yield raw
    raw.close()


def test_migration_is_idempotent(con):
    sql = _MIGRATION.read_text(encoding="utf-8")
    for statement in sql.split(";"):
        if statement.strip():
            con.execute(statement)
    assert store.load_view_records() == []


def test_save_and_load_one_row_per_day(con):
    store.save_view_record(ViewRecord(date(2026, 11, 18), "up", "claude", "down", None))
    store.save_view_record(ViewRecord(date(2026, 11, 17), "down", "fallback", "down", None))
    records = store.load_view_records()
    assert records == [
        ViewRecord(date(2026, 11, 17), "down", "fallback", "down", None),
        ViewRecord(date(2026, 11, 18), "up", "claude", "down", None),
    ]


def test_same_day_is_overwritten(con):
    day = date(2026, 11, 17)
    store.save_view_record(ViewRecord(day, "up", "claude", "up", None))
    store.save_view_record(ViewRecord(day, "down", "fallback", "down", None))
    assert store.load_view_records() == [ViewRecord(day, "down", "fallback", "down", None)]
    assert con.execute("SELECT COUNT(*) FROM morning_view_log").fetchone()[0] == 1


def test_record_outcome_fills_actual_change(con):
    day = date(2026, 11, 17)
    store.save_view_record(ViewRecord(day, "up", "claude", "down", None))
    assert store.record_view_outcome(day, 1.25) is True
    assert store.load_view_records() == [ViewRecord(day, "up", "claude", "down", 1.25)]
    judged_at = con.execute("SELECT judged_at FROM morning_view_log").fetchone()[0]
    assert judged_at is not None


def test_record_outcome_without_announcement_makes_no_row(con):
    assert store.record_view_outcome(date(2026, 11, 17), 0.5) is False
    assert con.execute("SELECT COUNT(*) FROM morning_view_log").fetchone()[0] == 0


def test_delete_removes_only_that_day(con):
    store.save_view_record(ViewRecord(date(2026, 11, 17), "up", "fallback", "up", None))
    store.save_view_record(ViewRecord(date(2026, 11, 18), "down", "fallback", "down", None))
    store.delete_view_record(date(2026, 11, 18))
    assert store.load_view_records() == [
        ViewRecord(date(2026, 11, 17), "up", "fallback", "up", None)
    ]


def test_loaded_records_feed_tally(con):
    # 手で数えた答え:
    # 11-10 up/claude/futures up,  +1.0 → 実際 up: 目線 勝, いつも上げ 勝, 先物 勝
    # 11-11 down/fallback/down,    -0.5 → 実際 down: 目線 勝, いつも上げ 負, 先物 勝
    # 11-12 up/claude/down,        -0.3 → 実際 down: 目線 負, いつも上げ 負, 先物 勝
    # 11-13 down/claude/up,         0.0 → 引き分け（数えない）
    # 11-14 up/fallback/up,        未答え合わせ（数えない）
    # 目線 2勝1敗, いつも上げ 1勝2敗, 先物 3勝0敗, 告知 5 日, fallback 2 日, 判定 3
    rows = [
        (date(2026, 11, 10), "up", "claude", "up", 1.0),
        (date(2026, 11, 11), "down", "fallback", "down", -0.5),
        (date(2026, 11, 12), "up", "claude", "down", -0.3),
        (date(2026, 11, 13), "down", "claude", "up", 0.0),
        (date(2026, 11, 14), "up", "fallback", "up", None),
    ]
    for day, view, source, fut, _ in rows:
        store.save_view_record(ViewRecord(day, view, source, fut, None))
    for day, *_, actual in rows:
        if actual is not None:
            assert store.record_view_outcome(day, actual) is True

    t = tally(store.load_view_records())
    assert (t.view.wins, t.view.losses) == (2, 1)
    assert (t.always_up.wins, t.always_up.losses) == (1, 2)
    assert (t.futures.wins, t.futures.losses) == (3, 0)
    assert t.announced_days == 5
    assert t.fallback_days == 2
    assert t.judged == 3
