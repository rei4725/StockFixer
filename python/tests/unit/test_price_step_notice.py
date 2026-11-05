"""段差の足で外した銘柄を、回に1通の知らせにまとめるテスト（#1643）"""

from typing import Optional
from unittest.mock import patch

import pytest
from tests.unit.test_daily_fetch_error_notice import _run_daily_pipeline

from src.domain.types import BatchFailure, BatchResult, MacroStaleSkip
from src.market_data.quality_check import QualityCheckResult, QualityIssue, price_step_issue
from src.orchestration.jobs.unavailable_gates import _raise_if_fetch_mostly_unavailable

_GATES = "src.orchestration.jobs.unavailable_gates"
_SEND = "src.reporting.discord.discord_utils.send_webhook_text"
_STEP_NOTICE = "段差"


def _skip(market: str, symbol: str, issue: Optional[QualityIssue] = None) -> dict:
    quality = None
    if issue is not None:
        quality = QualityCheckResult(market=market, symbol=symbol, passed=True, issues=[issue])
    return {"market": market, "symbol": symbol, "status": "skip", "quality_result": quality}


def _step_skip(market: str, symbol: str) -> dict:
    return _skip(market, symbol, price_step_issue([("2026-10-29", 1.8)]))


def _notices(batch: BatchResult, total: int = 100) -> list[tuple[str, str]]:
    with patch(f"{_GATES}._notify_stage_failure") as notify:
        _raise_if_fetch_mostly_unavailable(total, batch)
    return [(call.args[0], str(call.args[1])) for call in notify.call_args_list]


def test_two_step_skips_make_one_notice_with_count_and_symbols():
    """段差の2銘柄は1通にまとまり、足の止まった銘柄は数えない"""
    batch = BatchResult(
        succeeded=[object()] * 97,
        skipped=[_step_skip("us", "EPAM"), _skip("us", "PSKY"), _step_skip("jp", "3350")],
    )

    notices = _notices(batch)

    assert len(notices) == 1
    label, message = notices[0]
    assert _STEP_NOTICE in label
    assert "2 件" in message
    assert "対象 100" in message
    assert "us_EPAM" in message and "jp_3350" in message
    assert "PSKY" not in message


@pytest.mark.parametrize(
    "skipped",
    [
        [_skip("us", "PSKY"), _skip("us", "VYLR")],
        [_skip("us", "PSKY", QualityIssue("zero_volume", "warning", "出来高が0"))],
        [object(), object()],
    ],
    ids=["no-quality", "other-check", "not-a-dict"],
)
def test_skips_without_price_step_add_no_notice(skipped):
    """quality_result の無いスキップや段差でない品質の問題だけの回は、知らせが増えない"""
    batch = BatchResult(succeeded=[object()] * 98, skipped=skipped)

    assert _notices(batch) == []


def test_notice_lists_only_head_symbols_in_a_stable_order():
    """銘柄は先頭だけを並べる。取得の終わった順によらず、毎朝同じ並びになる"""
    symbols = ["S5", "S3", "S0", "S4", "S2", "S1"]
    batch = BatchResult(skipped=[_step_skip("us", s) for s in symbols])

    ((_, message),) = _notices(batch)

    assert "6 件" in message
    assert "us_S0" in message and "us_S4" in message
    assert "us_S5" not in message
    assert "ほか" in message


def test_other_notices_stay_as_they_are():
    """取得のエラー（#1256）と相場の指標（#1634）の知らせは、段差の知らせと別に今までどおり出る"""
    stale = MacroStaleSkip(market="us", symbol="A", columns={"vix_close": "2026-10-15"})
    failure = BatchFailure(market="us", symbol="B", error="接続取得がタイムアウト")
    batch = BatchResult(
        succeeded=[object()] * 96,
        failed=[failure],
        skipped=[_step_skip("us", "EPAM"), _skip("us", "A")],
        macro_stale=[stale],
    )

    notices = _notices(batch)

    assert [label for label, _ in notices] == [
        "[1/6] データ取得の一部の銘柄",
        "[1/6] データ取得の相場の指標",
        "[1/6] データ取得の段差の足",
    ]
    assert notices[0][1] == "取得のエラーが 1 件あった（対象 100）: us_B"
    assert "us_EPAM" not in notices[0][1] and "us_EPAM" not in notices[1][1]


def test_over_the_line_raises_without_step_notice():
    """線を超えた回は今までどおり止める文だけで、知らせを二重にしない"""
    batch = BatchResult(skipped=[_step_skip("us", f"S{i}") for i in range(6)])

    with patch(f"{_GATES}._notify_stage_failure") as notify:
        with pytest.raises(RuntimeError, match="多すぎる"):
            _raise_if_fetch_mostly_unavailable(10, batch)

    notify.assert_not_called()


def test_send_failure_does_not_stop_the_stage():
    """知らせの送信が落ちても [1/6] は例外を上げない"""
    batch = BatchResult(succeeded=[object()] * 9, skipped=[_step_skip("jp", "3350")])

    with patch(_SEND, side_effect=RuntimeError("送信できない")) as send:
        _raise_if_fetch_mostly_unavailable(10, batch)

    send.assert_called_once()


def test_daily_pipeline_sends_one_notice_and_continues():
    """日次の道でも知らせは1通で、後の段へ進む"""
    batch = BatchResult(succeeded=[object()] * 9, skipped=[_step_skip("jp", "3350")])

    mocks = _run_daily_pipeline(batch, total=10)

    assert mocks["raised"] is None
    mocks["webhook"].assert_called_once()
    assert "jp_3350" in mocks["webhook"].call_args.args[0]
    mocks["predict"].assert_called_once()
    mocks["error"].assert_not_called()
