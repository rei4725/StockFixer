"""[3/6] の予想が jp で大きく欠けた回に保存の印を立てないテスト（#1438）"""

from unittest.mock import MagicMock, patch

from src.domain.types import BatchResult, SymbolTask

_DAILY = "src.orchestration.jobs.daily"
_GATES = "src.orchestration.jobs.unavailable_gates"


def _rows(market: str, n: int) -> list[MagicMock]:
    return [MagicMock(market=market, symbol=f"{market}{i}") for i in range(n)]


def _tasks(n_jp: int, n_us: int) -> list[SymbolTask]:
    return [SymbolTask(market="jp", symbol=f"J{i}") for i in range(n_jp)] + [
        SymbolTask(market="us", symbol=f"U{i}") for i in range(n_us)
    ]


def _run(rows: list, tasks: list[SymbolTask]) -> dict:
    """日次の道を、予想の結果と対象を差し替えて流し、印と知らせの mock を返す。"""
    from tests.unit.test_daily_fetch_error_notice import _run_daily_pipeline

    batch = BatchResult(succeeded=[object()] * len(tasks), failed=[], skipped=[])
    with (
        patch(f"{_DAILY}.mark_prediction_saved") as mark,
        patch(f"{_DAILY}.clear_prediction_saved_marker"),
        patch(f"{_GATES}._notify_stage_failure") as notify,
    ):
        mocks = _run_daily_pipeline(batch, total=len(tasks), tasks=tasks, rows=rows)
    mocks["mark"] = mark
    mocks["notify"] = notify
    return mocks


def test_jp_predictions_mostly_missing_skips_marker_and_notifies_once():
    """jp 249 のうち 100 しか予想が出ない回は、印を立てずに jp の件数を1回知らせること"""
    mocks = _run(_rows("jp", 100) + _rows("us", 535), _tasks(249, 535))

    mocks["mark"].assert_not_called()
    mocks["notify"].assert_called_once()
    stage, error = mocks["notify"].call_args.args
    assert "[3/6]" in stage
    message = str(error)
    assert "jp" in message
    assert "149" in message
    assert "249" in message
    assert mocks["raised"] is None
    mocks["completion"].assert_called_once()


def test_us_only_missing_still_marks():
    """us だけが 300 欠けた回（jp は全部出た）は今までどおり印を立て、知らせないこと"""
    mocks = _run(_rows("jp", 249) + _rows("us", 235), _tasks(249, 535))

    mocks["mark"].assert_called_once()
    mocks["notify"].assert_not_called()


def test_jp_few_missing_within_line_still_marks():
    """jp の欠けが線の内（数銘柄）なら印を立てること"""
    mocks = _run(_rows("jp", 246) + _rows("us", 535), _tasks(249, 535))

    mocks["mark"].assert_called_once()
    mocks["notify"].assert_not_called()


def test_no_jp_targets_does_not_divide_by_zero():
    """jp の対象が0件でも0で割らず、印を立てること"""
    mocks = _run(_rows("us", 535), _tasks(0, 535))

    mocks["mark"].assert_called_once()
    mocks["notify"].assert_not_called()
