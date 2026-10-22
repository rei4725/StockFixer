"""src.orchestration.jobs.accuracy_json のテスト（#843）"""

from unittest.mock import MagicMock, patch

from src.orchestration.jobs.accuracy_json import (
    append_accuracy_json_sectors,
    write_accuracy_json_without_sectors,
)


@patch("src.prediction.accuracy_report.prepare_daily_accuracy_report")
def test_write_without_sectors_returns_report(mock_prepare):
    """業種なしで書き、書き足しに使う材料を返すこと"""
    report = mock_prepare.return_value

    assert write_accuracy_json_without_sectors() is report
    report.write.assert_called_once_with(sector_getter=None)
    report.write_with_sectors.assert_not_called()


@patch(
    "src.prediction.accuracy_report.prepare_daily_accuracy_report",
    side_effect=RuntimeError("disk full"),
)
def test_write_without_sectors_failure_returns_none(_mock_prepare):
    """書き出しが例外でも外へ出さず None を返すこと"""
    assert write_accuracy_json_without_sectors() is None


def test_append_sectors_skips_when_no_report():
    """材料が無ければ何もしないこと"""
    append_accuracy_json_sectors(None)


def test_append_sectors_writes_and_swallows_error():
    """業種を書き足し、失敗しても例外を外へ出さないこと"""
    report = MagicMock()
    append_accuracy_json_sectors(report)
    report.write_with_sectors.assert_called_once_with()

    report.write_with_sectors.side_effect = RuntimeError("429")
    append_accuracy_json_sectors(report)
