"""ユニットテスト: 週次ルール評価ジョブの失敗の扱い（#1087）"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.jobs.weekly import run_weekly_rule_evaluation

_EVAL = "src.backtest.rule_selector.evaluate_all_symbols"
_NOTIFY = "src.reporting.discord.discord_utils.send_rule_evaluation_completion"


def _summary(evaluated: int, effective: int, skipped: int, errors: int) -> dict[str, int]:
    return {"evaluated": evaluated, "effective": effective, "skipped": skipped, "errors": errors}


def test_errors_notify_then_raise() -> None:
    calls: list[str] = []
    notify = MagicMock(side_effect=lambda **_: calls.append("notify") or True)
    with (
        patch(_EVAL, return_value=_summary(3, 1, 0, 2)),
        patch(_NOTIFY, notify),
    ):
        with pytest.raises(RuntimeError, match="2"):
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
