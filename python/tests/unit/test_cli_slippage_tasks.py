"""run_backtest_optimize_batch.py / run_walk_forward_report.py の CLI 引数と呼び出しの形のテスト（Issue #1021）。

--slippage を省くと None（市場別の既定）、0 を明示すると 0.0 を渡すこと、
load_target_symbols で作った tasks を渡し、関数が受けない as_of_date を渡さないことを確かめる。
"""

from unittest.mock import patch

import pytest
import run_backtest_optimize_batch
import run_walk_forward_report

from src.domain.types import SymbolTask

_TASKS = [SymbolTask(market="us", symbol="AAPL"), SymbolTask(market="jp", symbol="7203")]


@pytest.mark.parametrize("module", [run_backtest_optimize_batch, run_walk_forward_report])
class TestParseArgsSlippage:
    def test_slippage_omitted_is_none(self, module, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog"])
        assert module.parse_args().slippage is None

    def test_slippage_zero_is_explicit_zero(self, module, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog", "--slippage", "0"])
        assert module.parse_args().slippage == 0.0


class TestOptimizeBatchMain:
    def _run(self, monkeypatch, argv):
        monkeypatch.setattr("sys.argv", ["prog", *argv])
        with (
            patch("src.orchestration.port_wiring.wire_ports"),
            patch.object(
                run_backtest_optimize_batch, "load_target_symbols", return_value=_TASKS
            ) as mock_load,
            patch.object(
                run_backtest_optimize_batch, "run_optimize_batch", return_value=[], autospec=True
            ) as mock_run,
        ):
            run_backtest_optimize_batch.main()
        return mock_load, mock_run

    def test_main_passes_tasks_and_none_slippage(self, monkeypatch):
        mock_load, mock_run = self._run(monkeypatch, [])

        mock_load.assert_called_once_with(None)
        args, kwargs = mock_run.call_args
        passed_tasks = args[0] if args else kwargs["tasks"]
        assert passed_tasks == _TASKS
        assert kwargs["slippage"] is None
        assert "as_of_date" not in kwargs

    def test_as_of_date_goes_to_load_target_symbols(self, monkeypatch):
        mock_load, _ = self._run(monkeypatch, ["--as-of-date", "2025-01-31"])

        mock_load.assert_called_once_with("2025-01-31")


class TestWalkForwardReportMain:
    _RESULT = {
        "summary_path": "s",
        "comparison_path": "c",
        "markdown_path": "m",
        "success": 2,
        "failed": 0,
        "total": 2,
    }

    def _run(self, monkeypatch, argv):
        monkeypatch.setattr("sys.argv", ["prog", *argv])
        with (
            patch("src.orchestration.port_wiring.wire_ports"),
            patch.object(
                run_walk_forward_report, "load_target_symbols", return_value=_TASKS
            ) as mock_load,
            patch.object(
                run_walk_forward_report,
                "run_walk_forward_comparison_report",
                return_value=self._RESULT,
                autospec=True,
            ) as mock_run,
        ):
            run_walk_forward_report.main()
        return mock_load, mock_run

    def test_main_passes_tasks_and_none_slippage(self, monkeypatch):
        mock_load, mock_run = self._run(monkeypatch, [])

        mock_load.assert_called_once_with(None)
        args, kwargs = mock_run.call_args
        passed_tasks = args[0] if args else kwargs["tasks"]
        assert passed_tasks == _TASKS
        assert kwargs["slippage"] is None
        assert "as_of_date" not in kwargs

    def test_as_of_date_goes_to_load_target_symbols(self, monkeypatch):
        mock_load, _ = self._run(monkeypatch, ["--as-of-date", "2025-01-31"])

        mock_load.assert_called_once_with("2025-01-31")
