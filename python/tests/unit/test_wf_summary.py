"""最新の Walk-Forward サマリーを読む共通の関数のテスト（#2145）。"""

import logging
from unittest.mock import patch

import pytest

from src.utils import wf_summary
from src.utils.wf_summary import load_latest_wf_summary


@pytest.fixture
def mock_logger():
    with patch.object(wf_summary, "logger", spec=logging.Logger) as mocked:
        yield mocked


def test_no_csv_returns_none_and_warns(tmp_path, mock_logger):
    (tmp_path / "other.csv").write_text("a\n1\n", encoding="utf-8")

    assert load_latest_wf_summary(tmp_path) == (None, None)
    mock_logger.warning.assert_called_once_with("Walk-Forward レポート CSV が見つかりません")
    mock_logger.error.assert_not_called()


def test_missing_dir_returns_none(tmp_path, mock_logger):
    assert load_latest_wf_summary(tmp_path / "missing") == (None, None)
    mock_logger.warning.assert_called_once_with("Walk-Forward レポート CSV が見つかりません")


def test_picks_newest_by_name_not_by_mtime(tmp_path, mock_logger):
    # 名前の新しい方を先に書き、更新時刻では古くする
    newer = tmp_path / "wf_summary_20260201_120000.csv"
    newer.write_text("symbol,sharpe\nAAPL,2.0\nMSFT,3.0\n", encoding="utf-8")
    older = tmp_path / "wf_summary_20260101_120000.csv"
    older.write_text("symbol,sharpe\nAAPL,1.0\n", encoding="utf-8")

    df, name = load_latest_wf_summary(tmp_path)

    assert name == "wf_summary_20260201_120000.csv"
    assert df["sharpe"].tolist() == [2.0, 3.0]
    mock_logger.info.assert_called_once_with(
        "WF サマリー読み込み: wf_summary_20260201_120000.csv (2 銘柄)"
    )


def test_broken_newest_csv_returns_none_without_fallback(tmp_path, mock_logger):
    (tmp_path / "wf_summary_20260101_120000.csv").write_text("symbol\nAAPL\n", encoding="utf-8")
    (tmp_path / "wf_summary_20260201_120000.csv").write_text("", encoding="utf-8")

    assert load_latest_wf_summary(tmp_path) == (None, None)
    mock_logger.error.assert_called_once()
    args, kwargs = mock_logger.error.call_args
    assert args[0].startswith("WF サマリー読み込み失敗: ")
    assert kwargs == {"exc_info": True}
