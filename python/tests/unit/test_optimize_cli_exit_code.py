"""手で流す最適化の CLI 2本の終了コード（#1466）。

最適パラメータを書けなかった銘柄があれば ERROR を出して終了コード 1 で終えることを確かめる。
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import run_backtest_optimize
import run_backtest_optimize_batch

from src.domain.types import SymbolTask


class TestOptimizeMainExitCode:
    def _run(self, monkeypatch, result_df: pd.DataFrame, **patches) -> MagicMock:
        monkeypatch.setattr("sys.argv", ["prog", "--market", "jp", "--symbol", "7203"])
        logger = MagicMock()
        with (
            patch("src.orchestration.port_wiring.wire_ports"),
            patch.object(run_backtest_optimize, "run_optimization", return_value=result_df),
            patch.object(run_backtest_optimize, "print_optimization_results"),
            patch.object(run_backtest_optimize, "save_optimization_results", return_value="r.csv"),
            patch.object(run_backtest_optimize, "logger", logger),
        ):
            for name, value in patches.items():
                monkeypatch.setattr(run_backtest_optimize, name, value)
            run_backtest_optimize.main()
        return logger

    def test_error_only_result_exits_one_without_saved_info(self, monkeypatch):
        error_df = pd.DataFrame([{"threshold": 0.0, "error": "boom"}])

        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, error_df)

        assert exc.value.code == 1

    def test_error_only_result_logs_error_not_saved_info(self, monkeypatch):
        error_df = pd.DataFrame([{"threshold": 0.0, "error": "boom"}])
        logger = MagicMock()
        monkeypatch.setattr("sys.argv", ["prog", "--market", "jp", "--symbol", "7203"])
        with (
            patch("src.orchestration.port_wiring.wire_ports"),
            patch.object(run_backtest_optimize, "run_optimization", return_value=error_df),
            patch.object(run_backtest_optimize, "print_optimization_results"),
            patch.object(run_backtest_optimize, "save_optimization_results", return_value="r.csv"),
            patch.object(run_backtest_optimize, "logger", logger),
            pytest.raises(SystemExit),
        ):
            run_backtest_optimize.main()

        logger.error.assert_called_once()
        info_messages = [str(c.args[0]) for c in logger.info.call_args_list]
        assert not any("最適パラメータ保存" in m for m in info_messages)

    def test_saved_params_returns_normally(self, monkeypatch):
        ok_df = pd.DataFrame([{"threshold": 0.0, "sharpe_ratio": 1.0}])

        logger = self._run(
            monkeypatch,
            ok_df,
            save_optimal_params_json=MagicMock(return_value="config/optimal_params.json"),
        )

        logger.error.assert_not_called()
        info_messages = [str(c.args[0]) for c in logger.info.call_args_list]
        assert any("最適パラメータ保存" in m for m in info_messages)


class TestOptimizeBatchMainExitCode:
    _TASKS = [SymbolTask(market="jp", symbol="7203"), SymbolTask(market="us", symbol="AAPL")]

    def _run(self, monkeypatch, results: list[dict]) -> MagicMock:
        monkeypatch.setattr("sys.argv", ["prog"])
        logger = MagicMock()
        with (
            patch("src.orchestration.port_wiring.wire_ports"),
            patch.object(
                run_backtest_optimize_batch, "load_target_symbols", return_value=self._TASKS
            ),
            patch.object(run_backtest_optimize_batch, "run_optimize_batch", return_value=results),
            patch.object(run_backtest_optimize_batch, "logger", logger),
        ):
            run_backtest_optimize_batch.main()
        return logger

    def test_one_error_exits_one(self, monkeypatch):
        results = [
            {"market": "jp", "symbol": "7203", "status": "success"},
            {"market": "us", "symbol": "AAPL", "status": "error", "error": "有効な結果なし"},
        ]

        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, results)

        assert exc.value.code == 1

    def test_no_error_returns_normally(self, monkeypatch):
        results = [
            {"market": "jp", "symbol": "7203", "status": "success"},
            {"market": "us", "symbol": "AAPL", "status": "skipped"},
        ]

        logger = self._run(monkeypatch, results)

        logger.error.assert_not_called()
