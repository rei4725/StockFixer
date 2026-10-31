"""run_rule_evaluate.py の終了コード・知らせ・既定の期間（#1459）。"""

from datetime import date
from unittest.mock import MagicMock, patch

import run_rule_evaluate


def _summary(errors: int) -> dict[str, int]:
    return {"evaluated": 10, "effective": 4, "skipped": 1, "errors": errors}


def _run(argv: list[str], errors: int) -> tuple[int, MagicMock, MagicMock]:
    evaluate = MagicMock(return_value=_summary(errors))
    notify = MagicMock()
    with patch("src.orchestration.port_wiring.wire_ports"), patch.object(
        run_rule_evaluate, "evaluate_all_symbols", evaluate
    ), patch.object(run_rule_evaluate, "print_rule_summary"), patch(
        "src.reporting.discord.discord_utils.send_rule_evaluation_completion", notify
    ), patch(
        "src.utils.japan_time.now_jst", return_value=MagicMock(date=lambda: date(2026, 11, 1))
    ), patch(
        "config.execution.RULE_EVAL_START", None
    ), patch(
        "config.execution.RULE_EVAL_END", None
    ), patch(
        "config.execution.RULE_EVAL_YEARS", 3
    ):
        rc = run_rule_evaluate.main(argv)
    return rc, evaluate, notify


def test_errors_exit_one_and_notify_errors() -> None:
    rc, _, notify = _run(["--market", "jp"], errors=2)

    assert rc == 1
    notify.assert_called_once()
    assert notify.call_args.kwargs["errors"] == 2


def test_no_errors_exit_zero() -> None:
    rc, _, notify = _run(["--market", "jp"], errors=0)

    assert rc == 0
    assert notify.call_args.kwargs["errors"] == 0


def test_default_window_matches_weekly_job() -> None:
    _, evaluate, _ = _run(["--market", "jp"], errors=0)

    kwargs = evaluate.call_args.kwargs
    assert kwargs["backtest_start"] == "2023-11-01"
    assert kwargs["backtest_end"] == "2026-11-01"


def test_explicit_window_is_used_as_is() -> None:
    _, evaluate, _ = _run(
        ["--market", "jp", "--start", "2021-01-01", "--end", "2025-01-01"], errors=0
    )

    kwargs = evaluate.call_args.kwargs
    assert kwargs["backtest_start"] == "2021-01-01"
    assert kwargs["backtest_end"] == "2025-01-01"


def test_no_discord_still_exits_one_on_errors() -> None:
    rc, _, notify = _run(["--market", "jp", "--no-discord"], errors=2)

    assert rc == 1
    notify.assert_not_called()


def test_show_only_does_not_evaluate() -> None:
    with patch("src.orchestration.port_wiring.wire_ports"), patch.object(
        run_rule_evaluate, "evaluate_all_symbols"
    ) as evaluate, patch.object(run_rule_evaluate, "print_rule_summary") as show:
        rc = run_rule_evaluate.main(["--market", "jp", "--show-only"])

    assert rc == 0
    show.assert_called_once_with("jp")
    evaluate.assert_not_called()


def test_missing_errors_key_treated_as_zero() -> None:
    evaluate = MagicMock(return_value={"evaluated": 1, "effective": 0, "skipped": 0})
    with patch("src.orchestration.port_wiring.wire_ports"), patch.object(
        run_rule_evaluate, "evaluate_all_symbols", evaluate
    ), patch.object(run_rule_evaluate, "print_rule_summary"):
        rc = run_rule_evaluate.main(["--market", "jp", "--no-discord"])

    assert rc == 0
