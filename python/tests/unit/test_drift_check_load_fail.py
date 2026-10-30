"""日次ドリフト検査が精度の読み込みの失敗を「精度データなし」と分けることのテスト（#1271）。"""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from tests.unit.models_dir_isolation import isolate_models_dir  # noqa: F401

from src.orchestration.jobs.drift_check import run_daily_drift_check
from src.prediction.db import load_drift_summary

_ACC = "src.prediction.db.accuracy"


@patch(f"{_ACC}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_ACC}.pd.read_sql", side_effect=RuntimeError("could not write to disk"))
@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
def test_read_failure_raises(mock_streak, mock_read_sql, mock_con):
    """read_sql が落ちた日は「精度データなし」で返らず、例外を投げる。"""
    with pytest.raises(RuntimeError, match="could not write to disk"):
        run_daily_drift_check()

    mock_streak.assert_not_called()


@patch(f"{_ACC}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_ACC}.pd.read_sql", return_value=pd.DataFrame())
@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
def test_no_rows_skips(mock_streak, mock_read_sql, mock_con):
    """採点の行が1件も無い日は今までどおり例外にならずスキップする。"""
    assert run_daily_drift_check() is None

    mock_streak.assert_not_called()


@patch(f"{_ACC}._db_connection", return_value=nullcontext(MagicMock()))
@patch(f"{_ACC}.pd.read_sql", side_effect=RuntimeError("db down"))
def test_other_callers_still_get_empty(mock_read_sql, mock_con):
    """既定の呼び方（レポート・run_accuracy_check）は今までどおり空の表で返る。"""
    assert load_drift_summary(horizon=1).empty
