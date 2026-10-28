"""ユニットテスト: upsert_stock_features の消すと書くを1つのトランザクションにする（#1340）

書き込みが落ちたら DELETE も戻し、その銘柄の前の行を残して、例外は呼び手へ投げる。
実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。テストの `_db_connection` は
with ごとに SAVEPOINT を切り、本番の自動確定と違って書きかけが戻ってしまうため、ここでは
SAVEPOINT を切らずに共有接続をそのまま渡す形に差し替える。
"""

from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.db import _connection
from src.utils.db import stock_features as sf


@pytest.fixture
def plain_connection(monkeypatch) -> None:
    """stock_features の接続を、自動確定の本番と同じく SAVEPOINT を切らない形にする"""

    @contextmanager
    def _plain(lock_timeout: float | None = None) -> Iterator[Any]:
        yield _connection._test_connection

    monkeypatch.setattr(sf, "_db_connection", _plain)


def _df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=len(closes), freq="B")
    return pd.DataFrame({"close": closes, "volume": [1000] * len(closes)}, index=idx)


def _rows(symbol: str) -> list[tuple]:
    with _connection._db_connection() as con:
        return con.execute(
            "SELECT row_num, date, close, volume FROM stock_features "
            "WHERE market = 'us' AND symbol = %s ORDER BY row_num",
            [symbol],
        ).fetchall()


def test_write_failure_keeps_previous_rows_and_raises(plain_connection):
    sf.upsert_stock_features("us", "AAPL", _df([100.0, 101.0, 102.0]))
    before = _rows("AAPL")
    with (
        patch.object(sf, "bulk_insert", side_effect=RuntimeError("COPY の失敗")),
        pytest.raises(RuntimeError, match="COPY の失敗"),
    ):
        sf.upsert_stock_features("us", "AAPL", _df([200.0]))
    assert len(before) == 3
    assert _rows("AAPL") == before


def test_write_success_replaces_rows_inside_transaction(plain_connection):
    """COPY がトランザクションの中で動き、行の中身と数は今までどおり置き換わる"""
    sf.upsert_stock_features("us", "AAPL", _df([100.0, 101.0, 102.0]))
    sf.upsert_stock_features("us", "MSFT", _df([50.0]))
    sf.upsert_stock_features("us", "AAPL", _df([200.0, 201.0]))
    rows = _rows("AAPL")
    assert [(r[0], float(r[2]), int(r[3])) for r in rows] == [(0, 200.0, 1000), (1, 201.0, 1000)]
    assert [pd.Timestamp(r[1]) for r in rows] == list(pd.date_range("2026-01-05", periods=2))
    assert len(_rows("MSFT")) == 1
