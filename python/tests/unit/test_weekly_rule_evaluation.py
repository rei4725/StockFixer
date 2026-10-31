"""ユニットテスト: 週次ルール評価ジョブの失敗の扱い（#1087）"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.jobs.weekly import run_weekly_rule_evaluation

_EVAL = "src.backtest.rule_selector.evaluate_all_symbols"
_NOTIFY = "src.reporting.discord.discord_utils.send_rule_evaluation_completion"
_SYMBOLS = "src.backtest.rule_selector._load_symbols"
_PORT = "src.backtest.rule_backtester.get_backtest_data_port"
_UPSERT = "src.backtest.rule_selector.upsert_rule_best"


def _record(calls: list[str]) -> bool:
    calls.append("notify")
    return True


def _summary(evaluated: int, effective: int, skipped: int, errors: int) -> dict[str, int]:
    return {"evaluated": evaluated, "effective": effective, "skipped": skipped, "errors": errors}


def test_errors_notify_then_raise() -> None:
    calls: list[str] = []
    notify = MagicMock(side_effect=lambda **_: _record(calls))
    with (
        patch(_EVAL, return_value=_summary(3, 1, 0, 2)),
        patch(_NOTIFY, notify),
    ):
        # やり直しても同じ失敗と同じ知らせが重なるだけなので、リトライさせない型で投げる
        with pytest.raises(NonRetryableJobError, match="2"):
            run_weekly_rule_evaluation()
    assert calls == ["notify"]
    assert notify.call_args.kwargs["errors"] == 2
    assert notify.call_args.kwargs["skipped"] == 0


def test_errors_raise_even_when_notify_fails() -> None:
    with (
        patch(_EVAL, return_value=_summary(1, 0, 0, 1)),
        patch(_NOTIFY, side_effect=RuntimeError("webhook down")),
    ):
        with pytest.raises(RuntimeError, match="1"):
            run_weekly_rule_evaluation()


def _port(frame: pd.DataFrame) -> MagicMock:
    port = MagicMock()
    port.download.return_value = frame
    port.add_technical_indicators.side_effect = lambda df: df
    return port


def test_all_fetch_empty_notify_then_raise() -> None:
    """全銘柄の値段の取得が空なら、知らせを送ってからジョブを失敗にする（#1270）。"""
    calls: list[str] = []
    notify = MagicMock(side_effect=lambda **_: _record(calls))
    with (
        patch(_SYMBOLS, return_value=["7203", "6758"]),
        patch(_PORT, return_value=_port(pd.DataFrame())),
        patch(_UPSERT) as upsert,
        patch(_NOTIFY, notify),
    ):
        with pytest.raises(RuntimeError, match="2"):
            run_weekly_rule_evaluation()
    assert calls == ["notify"]
    assert notify.call_args.kwargs["errors"] == 2
    assert notify.call_args.kwargs["skipped"] == 0
    upsert.assert_not_called()


def test_fetched_without_valid_rule_is_skipped_and_job_succeeds() -> None:
    """取得はできて有効ルールが無い銘柄は skipped に数え、ジョブは成功で終わる（#1270）。"""
    notify = MagicMock(return_value=True)
    with (
        patch(_SYMBOLS, return_value=["7203", "6758"]),
        patch(_PORT, return_value=_port(pd.DataFrame({"Close": [100.0, 101.0]}))),
        patch("src.backtest.rule_selector.ALL_RULES", []),
        patch(_UPSERT) as upsert,
        patch(_NOTIFY, notify),
    ):
        run_weekly_rule_evaluation()
    notify.assert_called_once()
    assert notify.call_args.kwargs["skipped"] == 2
    assert notify.call_args.kwargs["errors"] == 0
    upsert.assert_not_called()


def test_no_valid_rule_is_skipped_and_job_succeeds() -> None:
    notify = MagicMock(return_value=True)
    with (
        patch(_EVAL, return_value=_summary(2, 0, 2, 0)),
        patch(_NOTIFY, notify),
    ):
        run_weekly_rule_evaluation()
    notify.assert_called_once()
    assert notify.call_args.kwargs["skipped"] == 2
    assert notify.call_args.kwargs["errors"] == 0
