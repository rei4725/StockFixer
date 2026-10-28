"""ユニットテスト: save_prediction_results の消すと書くを1つのトランザクションにする（#1354）

書き込みが落ちたら DELETE も戻し、同じ market・symbol・model_version の前の行を残して、
例外は呼び手へ投げる。実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。
テストの `_db_connection` は with ごとに SAVEPOINT を切り、本番の自動確定と違って書きかけが
戻ってしまうため、ここでは SAVEPOINT を切らずに共有接続をそのまま渡す形に差し替える。
"""

from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from src.prediction.db import prediction_results as pr
from src.prediction.types import PredictionResult
from src.utils.db import _connection


@pytest.fixture
def plain_connection(monkeypatch) -> None:
    """prediction_results の接続を、自動確定の本番と同じく SAVEPOINT を切らない形にする"""

    @contextmanager
    def _plain(lock_timeout: float | None = None) -> Iterator[Any]:
        yield _connection._test_connection

    monkeypatch.setattr(pr, "_db_connection", _plain)


def _rows(symbol: str, model_version: str = "production") -> list[tuple]:
    with _connection._db_connection() as con:
        return con.execute(
            "SELECT predicted_at, current_price FROM prediction_results "
            "WHERE market = 'us' AND symbol = %s AND model_version = %s ORDER BY predicted_at",
            [symbol, model_version],
        ).fetchall()


def test_write_failure_keeps_previous_rows_and_raises(plain_connection):
    pr.save_prediction_results(
        "20261028_070000",
        [
            PredictionResult("us", "AAPL", 150.0, 153.0, 0.02, 2),
            PredictionResult("us", "MSFT", 300.0, 306.0, 0.02, 2),
        ],
    )
    before_aapl = _rows("AAPL")
    before_msft = _rows("MSFT")
    with (
        patch.object(pr, "bulk_insert", side_effect=RuntimeError("COPY の失敗")),
        pytest.raises(RuntimeError, match="COPY の失敗"),
    ):
        pr.save_prediction_results(
            "20261029_070000", [PredictionResult("us", "AAPL", 160.0, 163.0, 0.02, 2)]
        )
    assert len(before_aapl) == 1
    assert _rows("AAPL") == before_aapl
    assert _rows("MSFT") == before_msft


def test_write_success_replaces_rows_inside_transaction(plain_connection):
    """書き込みがトランザクションの中で動き、同じ銘柄・版の行は今までどおり置き換わる"""
    pr.save_prediction_results(
        "20261028_070000",
        [
            PredictionResult("us", "AAPL", 150.0, 153.0, 0.02, 2),
            PredictionResult("us", "MSFT", 300.0, 306.0, 0.02, 2),
        ],
    )
    pr.save_prediction_results(
        "20261029_070000",
        [PredictionResult("us", "AAPL", 160.0, 163.0, 0.02, 2, model_version="challenger")],
    )
    pr.save_prediction_results(
        "20261029_070000", [PredictionResult("us", "AAPL", 160.0, 163.0, 0.02, 2)]
    )
    rows = _rows("AAPL")
    assert [(r[0], float(r[1])) for r in rows] == [("20261029_070000", 160.0)]
    assert len(_rows("AAPL", "challenger")) == 1
    assert len(_rows("MSFT")) == 1
