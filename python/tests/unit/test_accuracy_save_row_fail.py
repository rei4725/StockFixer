"""採点の行の書き込みに落ちた行があれば、ほかの行を書いた後に例外を投げることのテスト（#1371）。"""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from src.prediction.db.accuracy import save_prediction_accuracy

_ACC = "src.prediction.db.accuracy"


def _row(symbol: str) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "model_name": "production",
        "predicted_at": "20261028_220000",
        "horizon": 1,
        "predicted_price": 101.0,
        "actual_price": 102.0,
        "predicted_ratio": 0.01,
        "actual_ratio": 0.02,
        "direction_match": True,
    }


def _con_failing_on(*symbols: str) -> MagicMock:
    con = MagicMock()

    def execute(_sql, params):
        if params[1] in symbols:
            raise RuntimeError(f"disk full {params[1]}")

    con.execute.side_effect = execute
    return con


def test_partial_failure_writes_others_and_raises():
    """一部の行が落ちても、ほかの行は書き、落ちた件数と最初の例外を載せて投げる。"""
    con = _con_failing_on("7203", "9984")
    rows = [_row("6758"), _row("7203"), _row("8306"), _row("9984")]

    with patch(f"{_ACC}._db_connection", return_value=nullcontext(con)):
        with pytest.raises(RuntimeError, match="2件") as excinfo:
            save_prediction_accuracy(rows)

    assert con.execute.call_count == 4
    assert "disk full 7203" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_all_rows_written_returns_count():
    """全部書けたときは件数を返す。"""
    con = _con_failing_on()
    with patch(f"{_ACC}._db_connection", return_value=nullcontext(con)):
        assert save_prediction_accuracy([_row("6758"), _row("7203")]) == 2


def test_empty_rows_returns_zero():
    """空の行では接続を開かずに0を返す。"""
    with patch(f"{_ACC}._db_connection") as mock_con:
        assert save_prediction_accuracy([]) == 0
    mock_con.assert_not_called()
