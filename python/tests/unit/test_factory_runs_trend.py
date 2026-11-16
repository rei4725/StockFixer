"""戦略ファクトリーの台帳 factory_runs を読み返す表のテスト（#1907）。"""

from contextlib import contextmanager
from datetime import datetime

import pytest

import src.utils.db.factory_runs as factory_runs
from src.reporting.factory_runs_trend import (
    REASON_CATEGORIES,
    build_factory_runs_trend,
    classify_reason,
    format_factory_runs_trend_markdown,
    split_gate_reasons,
)
from src.utils.db.factory_runs import FactoryRunRecord, load_factory_runs

# factory_gate.py の今の文の形（値は例）
_NUM_TRADES = "num_trades 12 < 30"
_EFFECTIVE = "effective_symbols 2 < 5"
_DSR = "dsr 0.412 < 0.95"
_DSR_NAN = "dsr nan < 0.95"
_PORTFOLIO_DD = "portfolio_max_drawdown -0.312 < -0.25"
_WORST_DD = "max_drawdown -0.402 < -0.25"
_CHAMPION_Z = "portfolio_sharpe 0.812 vs champion 0.905: 差の z=-0.41 <= 1.0"
_CHAMPION_UNTESTABLE = "チャンピオンとの Sharpe 差を検定できない（日次リターンが無い）"
_NO_CHAMPION = "チャンピオンが無い（対照群が全滅しチャンピオン比較不能）ため不合格"


@pytest.mark.parametrize(
    "reason, expected",
    [
        (_NUM_TRADES, "num_trades"),
        (_EFFECTIVE, "effective_symbols"),
        (_DSR, "dsr"),
        (_DSR_NAN, "dsr"),
        (_PORTFOLIO_DD, "max_drawdown"),
        (_WORST_DD, "max_drawdown"),
        (_CHAMPION_Z, "champion_z"),
        (_CHAMPION_UNTESTABLE, "champion_z"),
        (_NO_CHAMPION, "no_champion"),
        ("pbo 0.8 > 0.5", "other"),
        ("", "other"),
    ],
)
def test_classify_reason(reason, expected):
    assert classify_reason(reason) == expected


def test_split_gate_reasons_joined_and_null():
    joined = "; ".join([_NUM_TRADES, _DSR, _PORTFOLIO_DD, _NO_CHAMPION, "謎の理由"])
    assert split_gate_reasons(joined) == [
        "num_trades",
        "dsr",
        "max_drawdown",
        "no_champion",
        "other",
    ]
    assert split_gate_reasons(None) == []
    assert split_gate_reasons("") == []
    # 同じ理由が1件に2度出ても1つ
    assert split_gate_reasons(f"{_PORTFOLIO_DD}; {_WORST_DD}") == ["max_drawdown"]


def _record(day, market, passed, reasons, hour=1):
    return FactoryRunRecord(
        hypothesis_hash=f"{day}-{market}-{hour}",
        market=market,
        evaluated_at=datetime.fromisoformat(f"{day}T{hour:02d}:00:00"),
        gate_passed=passed,
        gate_reasons=reasons,
        portfolio_sharpe_ratio=None,
        dsr=None,
        num_trades=None,
        portfolio_max_drawdown=None,
    )


def test_build_trend_counts_by_day_and_market():
    records = [
        _record("2026-11-12", "jp", False, f"{_DSR}; {_CHAMPION_Z}", hour=1),
        _record("2026-11-12", "jp", False, f"{_DSR}; {_EFFECTIVE}", hour=2),
        _record("2026-11-12", "jp", True, None, hour=3),
        _record("2026-11-11", "us", False, _NO_CHAMPION),
        _record("2026-11-11", "us", False, None, hour=2),
        _record("2026-11-11", "us", False, "未知の形", hour=3),
    ]
    df = build_factory_runs_trend(records)
    assert list(df.columns) == ["date", "market", "evaluated", "passed"] + [
        f"failed_{name}" for name in REASON_CATEGORIES
    ]
    assert list(zip(df["date"], df["market"])) == [("2026-11-11", "us"), ("2026-11-12", "jp")]

    us = df.iloc[0]
    assert us["evaluated"] == 3
    assert us["passed"] == 0
    assert us["failed_no_champion"] == 1
    # 不合格で理由が null の行と、知らない形の行は other
    assert us["failed_other"] == 2

    jp = df.iloc[1]
    assert jp["evaluated"] == 3
    assert jp["passed"] == 1
    assert jp["failed_dsr"] == 2
    assert jp["failed_champion_z"] == 1
    assert jp["failed_effective_symbols"] == 1
    assert jp["failed_num_trades"] == 0
    assert jp["failed_other"] == 0

    md = format_factory_runs_trend_markdown(df)
    assert md.splitlines()[2] == "| 2026-11-11 | us | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 |"


def test_empty_trend_formats_as_no_data():
    df = build_factory_runs_trend([])
    assert df.empty
    assert format_factory_runs_trend_markdown(df) == "データなし"


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

    monkeypatch.setattr(factory_runs, "_db_connection", fake_connection)
    return log


def test_load_factory_runs_reads_after_ensure_and_only_selects(monkeypatch):
    when = datetime(2026, 11, 12, 1, 0, 0)
    rows = [("h1", "jp", when, False, _DSR, 0.5, 0.4, 12, -0.1)]
    log = _patch_connection(monkeypatch, rows)

    records = load_factory_runs(market="jp", since=datetime(2026, 11, 1))

    statements = [sql for sql, _ in log]
    reads = [sql for sql in statements if not sql.upper().startswith(("CREATE", "ALTER"))]
    # ensure_factory_tables の DDL が先で、読むのは最後の SELECT 1文だけ
    assert statements[0].upper().startswith("CREATE TABLE IF NOT EXISTS FACTORY_RUNS")
    assert len(reads) == 1 and statements[-1] == reads[0]
    assert reads[0].upper().startswith("SELECT")
    for word in ("INSERT", "UPDATE", "DELETE"):
        assert word not in reads[0].upper()
    assert log[-1][1] == ["jp", datetime(2026, 11, 1)]

    assert records == [
        FactoryRunRecord(
            hypothesis_hash="h1",
            market="jp",
            evaluated_at=when,
            gate_passed=False,
            gate_reasons=_DSR,
            portfolio_sharpe_ratio=0.5,
            dsr=0.4,
            num_trades=12,
            portfolio_max_drawdown=-0.1,
        )
    ]


def test_load_factory_runs_without_filters(monkeypatch):
    log = _patch_connection(monkeypatch, [])
    assert load_factory_runs() == []
    assert "WHERE" not in log[-1][0].upper()
    assert log[-1][1] == []
