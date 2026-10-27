"""採点が予測結果の読み込みの失敗を「予測結果なし」と分けることのテスト（#1282）。"""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.prediction.db import load_prediction_results
from src.prediction.prediction_pipeline import run_accuracy_check

_PR = "src.prediction.db.prediction_results"


@patch(f"{_PR}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_PR}.pd.read_sql", side_effect=RuntimeError("could not write to disk"))
@patch("src.prediction.db.save_prediction_accuracy")
def test_read_failure_raises(mock_save, mock_read_sql, mock_con):
    """read_sql が落ちた日は「予測結果が存在しません」で返らず、例外を投げる。"""
    with pytest.raises(RuntimeError, match="could not write to disk"):
        run_accuracy_check(horizon=1)

    mock_save.assert_not_called()


@patch(f"{_PR}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_PR}.pd.read_sql", return_value=pd.DataFrame())
@patch("src.prediction.db.save_prediction_accuracy")
def test_no_rows_returns_empty(mock_save, mock_read_sql, mock_con):
    """予測結果の行が1件も無い日は今までどおり例外にならず空の表を返す。"""
    assert run_accuracy_check(horizon=1).empty

    mock_save.assert_not_called()


@patch(f"{_PR}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_PR}.pd.read_sql", side_effect=RuntimeError("db down"))
def test_default_callers_still_get_empty_with_limit(mock_read_sql, mock_con):
    """既定の呼び方は limit の道でも今までどおり空の表で返る。"""
    assert load_prediction_results(limit=10).empty


@patch(f"{_PR}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_PR}.pd.read_sql", side_effect=RuntimeError("db down"))
def test_default_callers_still_get_empty_with_predicted_at(mock_read_sql, mock_con):
    """既定の呼び方は predicted_at の道でも今までどおり空の表で返る。"""
    assert load_prediction_results(predicted_at="2026-10-27 00:00:00").empty


@patch(f"{_PR}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_PR}.pd.read_sql", side_effect=RuntimeError("db down"))
def test_raise_on_error_with_predicted_at(mock_read_sql, mock_con):
    """raise_on_error=True なら predicted_at の道でも例外を投げる。"""
    with pytest.raises(RuntimeError, match="db down"):
        load_prediction_results(predicted_at="2026-10-27 00:00:00", raise_on_error=True)
