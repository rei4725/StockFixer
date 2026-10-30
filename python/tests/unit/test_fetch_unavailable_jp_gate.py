"""[1/6] の取得できなかった銘柄の線を jp の銘柄だけでも見るテスト（#1432）"""

from unittest.mock import patch

import pytest

from src.domain.types import BatchFailure, BatchResult, SymbolTask

_GATES = "src.orchestration.jobs.unavailable_gates"


def _failures(market: str, n: int) -> list[BatchFailure]:
    return [BatchFailure(market=market, symbol=f"S{i:03d}", error="取得元の不調") for i in range(n)]


def _call(batch_result: BatchResult, market_totals: dict[str, int]) -> None:
    from src.orchestration.jobs.unavailable_gates import _raise_if_fetch_mostly_unavailable

    total = sum(market_totals.values())
    with patch(f"{_GATES}._notify_stage_failure"):
        _raise_if_fetch_mostly_unavailable(total, batch_result, market_totals=market_totals)


def test_jp_mostly_failed_raises_even_below_total_line():
    """jp 249・us 535 で jp が 200 件落ち、全体が約26%でも投げ、jp の件数と対象が出ること"""
    batch = BatchResult(succeeded=[object()] * 584, failed=_failures("jp", 200), skipped=[])

    with pytest.raises(RuntimeError) as exc:
        _call(batch, {"jp": 249, "us": 535})

    message = str(exc.value)
    assert "多すぎる" in message
    assert "jp" in message
    assert "200" in message
    assert "249" in message


def test_jp_skipped_count_toward_jp_line():
    """jp のスキップも jp の割合に数えること"""
    skipped = [{"market": "jp", "symbol": f"J{i}", "status": "skip"} for i in range(130)]
    batch = BatchResult(succeeded=[object()] * 654, failed=[], skipped=skipped)

    with pytest.raises(RuntimeError, match="jp"):
        _call(batch, {"jp": 249, "us": 535})


def test_us_only_failed_below_total_line_does_not_raise():
    """us だけが 300 件落ちた朝（全体 38%・jp 0%）は今までどおり投げないこと"""
    batch = BatchResult(
        succeeded=[object()] * 483,
        failed=_failures("us", 300),
        skipped=[{"market": "us", "symbol": "X", "status": "skip"}],
    )

    _call(batch, {"jp": 249, "us": 535})


def test_daily_stops_when_jp_mostly_failed():
    """日次の道が jp と us の対象を渡し、jp が大きく欠けた朝は CRITICAL で止まること"""
    from tests.unit.test_daily_fetch_error_notice import _run_daily_pipeline

    batch = BatchResult(succeeded=[object()] * 584, failed=_failures("jp", 200), skipped=[])
    tasks = [SymbolTask(market="jp", symbol=f"J{i}") for i in range(249)] + [
        SymbolTask(market="us", symbol=f"U{i}") for i in range(535)
    ]

    mocks = _run_daily_pipeline(batch, total=784, tasks=tasks)

    assert isinstance(mocks["raised"], RuntimeError)
    assert "jp" in str(mocks["raised"])
    mocks["error"].assert_called_once()
    mocks["predict"].assert_not_called()


def test_no_jp_targets_does_not_divide_by_zero():
    """jp の対象が0件でも0で割らないこと"""
    batch = BatchResult(succeeded=[object()] * 535, failed=[], skipped=[])

    _call(batch, {"us": 535})
    _call(batch, {"jp": 0, "us": 535})
