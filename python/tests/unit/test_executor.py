"""ユニットテスト: PipelineExecutor インターフェースと PipelineOrchestrator DI"""

from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.executor import (
    BacktestOptimizationExecutor,
    DataPipelineExecutor,
    ModelTrainingExecutor,
    PipelineExecutor,
    PipelineOrchestrator,
    PredictionExecutor,
    WatchlistRefreshExecutor,
)

# ──────────────────────────────────────────────
# PipelineExecutor Protocol の型チェック
# ──────────────────────────────────────────────


class TestPipelineExecutorProtocol:
    def test_concrete_executors_satisfy_protocol(self):
        """具体 Executor クラスが PipelineExecutor Protocol を満たすこと"""
        assert isinstance(DataPipelineExecutor(), PipelineExecutor)
        assert isinstance(PredictionExecutor(), PipelineExecutor)
        assert isinstance(ModelTrainingExecutor("XGBoostModel", "production"), PipelineExecutor)
        assert isinstance(BacktestOptimizationExecutor(), PipelineExecutor)
        assert isinstance(WatchlistRefreshExecutor(), PipelineExecutor)

    def test_stub_executor_satisfies_protocol(self):
        """name と execute を持つスタブクラスが Protocol を満たすこと"""

        class StubExecutor:
            @property
            def name(self) -> str:
                return "stub"

            def execute(self) -> None:
                pass

        assert isinstance(StubExecutor(), PipelineExecutor)

    def test_class_without_execute_does_not_satisfy_protocol(self):
        """execute メソッドを持たないオブジェクトは Protocol を満たさないこと"""

        class NoExecute:
            @property
            def name(self) -> str:
                return "no_execute"

        assert not isinstance(NoExecute(), PipelineExecutor)


# ──────────────────────────────────────────────
# PipelineOrchestrator の DI テスト
# ──────────────────────────────────────────────


class TestPipelineOrchestrator:
    def _make_mock_executor(self, name: str) -> MagicMock:
        mock = MagicMock()
        mock.name = name
        return mock

    def test_runs_critical_executors_in_order(self):
        """critical_executors が登録順に実行されること"""
        ex1 = self._make_mock_executor("step1")
        ex2 = self._make_mock_executor("step2")
        orchestrator = PipelineOrchestrator(critical_executors=[ex1, ex2])
        orchestrator.run()
        ex1.execute.assert_called_once()
        ex2.execute.assert_called_once()

    def test_runs_noncritical_executors_after_critical(self):
        """noncritical_executors は critical_executors の後に実行されること"""
        call_order = []
        critical = self._make_mock_executor("critical")
        critical.execute.side_effect = lambda: call_order.append("critical")
        noncritical = self._make_mock_executor("noncritical")
        noncritical.execute.side_effect = lambda: call_order.append("noncritical")

        orchestrator = PipelineOrchestrator(
            critical_executors=[critical],
            noncritical_executors=[noncritical],
        )
        orchestrator.run()

        assert call_order == ["critical", "noncritical"]

    def test_critical_failure_propagates_and_stops_pipeline(self):
        """critical_executor が失敗すると例外が伝播し後続 Executor が実行されないこと"""
        critical_fail = self._make_mock_executor("critical_fail")
        critical_fail.execute.side_effect = RuntimeError("致命的エラー")
        next_critical = self._make_mock_executor("next_critical")
        noncritical = self._make_mock_executor("noncritical")

        orchestrator = PipelineOrchestrator(
            critical_executors=[critical_fail, next_critical],
            noncritical_executors=[noncritical],
        )

        with pytest.raises(RuntimeError, match="致命的エラー"):
            orchestrator.run()

        next_critical.execute.assert_not_called()
        noncritical.execute.assert_not_called()

    def test_noncritical_failure_does_not_propagate(self):
        """noncritical_executor が失敗しても後続処理が継続されること"""
        noncritical_fail = self._make_mock_executor("noncritical_fail")
        noncritical_fail.execute.side_effect = Exception("非致命的エラー")
        noncritical_next = self._make_mock_executor("noncritical_next")

        orchestrator = PipelineOrchestrator(
            noncritical_executors=[noncritical_fail, noncritical_next],
        )
        orchestrator.run()  # 例外が外に出ないこと

        noncritical_next.execute.assert_called_once()

    def test_empty_executor_lists_runs_without_error(self):
        """空のリストを渡しても正常終了すること"""
        orchestrator = PipelineOrchestrator()
        orchestrator.run()  # 例外なし

    def test_mock_executors_can_be_injected(self):
        """モック Executor を注入して呼び出し引数を検証できること"""
        mock_data = self._make_mock_executor("data_pipeline")
        mock_predict = self._make_mock_executor("prediction_production")

        orchestrator = PipelineOrchestrator(
            critical_executors=[mock_data, mock_predict],
        )
        orchestrator.run()

        mock_data.execute.assert_called_once_with()
        mock_predict.execute.assert_called_once_with()


# ──────────────────────────────────────────────
# 具体 Executor の name プロパティテスト
# ──────────────────────────────────────────────


class TestExecutorNames:
    def test_data_pipeline_executor_name(self):
        assert DataPipelineExecutor().name == "data_pipeline"

    def test_prediction_executor_name_default(self):
        assert PredictionExecutor().name == "prediction_production"

    def test_prediction_executor_name_challenger(self):
        assert PredictionExecutor("challenger").name == "prediction_challenger"

    def test_model_training_executor_name(self):
        ex = ModelTrainingExecutor(model_type="XGBoostModel", model_name="challenger_xgb")
        assert ex.name == "training_challenger_xgb"

    def test_backtest_optimization_executor_name(self):
        assert BacktestOptimizationExecutor().name == "backtest_optimization"

    def test_watchlist_refresh_executor_name(self):
        assert WatchlistRefreshExecutor().name == "watchlist_refresh"


# ──────────────────────────────────────────────
# 具体 Executor の execute() 委譲テスト
# ──────────────────────────────────────────────


class TestDataPipelineExecutor:
    @patch("src.market_data.pipeline.run_batch_pipeline")
    @patch("src.watchlist.batch_runner.load_target_symbols")
    def test_delegates_to_run_batch_pipeline(self, mock_load_symbols, mock_run):
        mock_load_symbols.return_value = []
        DataPipelineExecutor().execute()
        mock_run.assert_called_once_with([])


class TestPredictionExecutor:
    @patch("src.prediction.prediction_pipeline.output_top_worst_results")
    @patch("src.prediction.prediction_pipeline.predict_all_unified")
    def test_delegates_to_predict_all_unified(self, mock_predict, mock_output):
        mock_predict.return_value = []
        PredictionExecutor().execute()
        mock_predict.assert_called_once_with()
        mock_output.assert_called_once_with(
            [],
            mode="unified",
            shadow_mode=True,
            model_version="production",
        )

    @patch("src.prediction.prediction_pipeline.output_top_worst_results")
    @patch("src.prediction.prediction_pipeline.predict_all_unified")
    def test_challenger_version_passed_to_output(self, mock_predict, mock_output):
        mock_predict.return_value = []
        PredictionExecutor("challenger").execute()
        mock_output.assert_called_once_with(
            [],
            mode="unified",
            shadow_mode=True,
            model_version="challenger",
        )


class TestModelTrainingExecutor:
    @patch("src.prediction.unified_model_pipeline.train_unified_model")
    def test_delegates_to_train_unified_model(self, mock_train):
        ModelTrainingExecutor("XGBoostModel", "production").execute()
        mock_train.assert_called_once_with(model_type="XGBoostModel", model_name="production")


class TestWatchlistRefreshExecutor:
    @patch("src.watchlist.manager.run_watchlist_refresh")
    def test_delegates_to_run_watchlist_refresh(self, mock_refresh):
        WatchlistRefreshExecutor().execute()
        mock_refresh.assert_called_once_with()


class TestBacktestOptimizationExecutor:
    @patch("src.backtest.optimizer.run_optimize_batch")
    def test_uses_grid_search_by_default(self, mock_run, monkeypatch):
        monkeypatch.delenv("USE_OPTUNA", raising=False)
        BacktestOptimizationExecutor().execute()
        mock_run.assert_called_once()

    @patch("src.backtest.optimizer.run_optuna_batch")
    def test_uses_optuna_when_env_set(self, mock_run, monkeypatch):
        monkeypatch.setenv("USE_OPTUNA", "true")
        BacktestOptimizationExecutor().execute()
        mock_run.assert_called_once()
