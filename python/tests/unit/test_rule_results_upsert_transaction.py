"""ユニットテスト: upsert_rule_best と upsert_rule_signal の消すと書くを1つのトランザクションにする（#2078）

INSERT が落ちたら DELETE も戻し、前の行を残して、例外は呼び手へ投げる。
実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。テストの `_db_connection` は
with ごとに SAVEPOINT を切り、本番の自動確定と違って書きかけが戻ってしまうため、ここでは
SAVEPOINT を切らずに共有接続をそのまま渡す形に差し替える（#1340 のテストと同じ形）。
"""

from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from src.utils.db import _connection
from src.utils.db import rule_results as rr


class _InsertFailingConnection:
    """INSERT の文だけを落とし、ほかは共有接続へそのまま渡す"""

    def __init__(self, con: Any, fail: bool) -> None:
        self._con = con
        self.fail = fail

    def execute(self, sql: str, params: Any = None) -> Any:
        if self.fail and "INSERT INTO" in sql:
            raise RuntimeError("INSERT の失敗")
        return self._con.execute(sql, params)

    def transaction(self) -> Any:
        return self._con.transaction()


@pytest.fixture
def failing_connection(monkeypatch) -> _InsertFailingConnection:
    proxy = _InsertFailingConnection(_connection._test_connection, fail=False)

    @contextmanager
    def _plain(lock_timeout: float | None = None) -> Iterator[Any]:
        yield proxy

    monkeypatch.setattr(rr, "_db_connection", _plain)
    return proxy


def _best_rows(symbol: str) -> list[tuple]:
    with _connection._db_connection() as con:
        return con.execute(
            "SELECT best_rule, win_rate FROM rule_best_by_symbol"
            " WHERE market = 'us' AND symbol = %s",
            [symbol],
        ).fetchall()


def _signal_rows(symbol: str) -> list[tuple]:
    with _connection._db_connection() as con:
        return con.execute(
            "SELECT rule_name, signal FROM rule_daily_signals"
            " WHERE signal_date = '2026-11-20' AND market = 'us' AND symbol = %s",
            [symbol],
        ).fetchall()


def _upsert_best(best_rule: str, win_rate: float) -> None:
    rr.upsert_rule_best(
        market="us",
        symbol="TXBEST",
        best_rule=best_rule,
        win_rate=win_rate,
        net_profit=1000.0,
        num_trades=10,
        profit_factor=1.5,
        max_drawdown=-0.2,
        backtest_start="2026-05-01",
        backtest_end="2026-11-14",
    )


def _upsert_signal(rule_name: str, signal: int) -> None:
    rr.upsert_rule_signal(
        signal_date="2026-11-20",
        market="us",
        symbol="TXSIG",
        rule_name=rule_name,
        signal=signal,
        price=150.0,
    )


def test_upsert_rule_best_insert_failure_keeps_previous_row_and_raises(failing_connection):
    _upsert_best("golden_cross", 0.6)
    before = _best_rows("TXBEST")
    failing_connection.fail = True
    with pytest.raises(RuntimeError, match="INSERT の失敗"):
        _upsert_best("rsi_reversal", 0.8)
    assert before == [("golden_cross", 0.6)]
    assert _best_rows("TXBEST") == before


def test_upsert_rule_best_success_replaces_row(failing_connection):
    _upsert_best("golden_cross", 0.6)
    _upsert_best("rsi_reversal", 0.8)
    assert _best_rows("TXBEST") == [("rsi_reversal", 0.8)]


def test_upsert_rule_signal_insert_failure_keeps_previous_row_and_raises(failing_connection):
    _upsert_signal("golden_cross", 1)
    before = _signal_rows("TXSIG")
    failing_connection.fail = True
    with pytest.raises(RuntimeError, match="INSERT の失敗"):
        _upsert_signal("rsi_reversal", -1)
    assert before == [("golden_cross", 1)]
    assert _signal_rows("TXSIG") == before


def test_upsert_rule_signal_success_replaces_row(failing_connection):
    _upsert_signal("golden_cross", 1)
    _upsert_signal("rsi_reversal", -1)
    assert _signal_rows("TXSIG") == [("rsi_reversal", -1)]
