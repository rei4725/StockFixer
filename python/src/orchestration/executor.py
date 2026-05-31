"""
PipelineExecutor インターフェースと BC 層ごとの具体実装。

スケジューラーへの DI 単位として機能し、テスト時にモック Executor を注入できる。
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from src.utils.logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class PipelineExecutor(Protocol):
    """パイプライン実行インターフェース。スケジューラーへの DI 単位。"""

    @property
    def name(self) -> str:
        """Executor の識別名。ログ・エラー報告に使用する。"""
        ...

    def execute(self) -> None:
        """パイプラインを実行する。失敗時は例外を送出する。"""
        ...


class DataPipelineExecutor:
    """market_data BC: 全マーケットのデータ取得パイプライン。"""

    @property
    def name(self) -> str:
        return "data_pipeline"

    def execute(self) -> None:
        from src.market_data.pipeline import run_batch_pipeline
        from src.watchlist.batch_runner import load_target_symbols

        tasks = load_target_symbols()
        run_batch_pipeline(tasks)


class PredictionExecutor:
    """prediction BC: 予測パイプライン（Top10/Worst10 出力）。"""

    def __init__(self, model_version: str = "production") -> None:
        self._model_version = model_version

    @property
    def name(self) -> str:
        return f"prediction_{self._model_version}"

    def execute(self) -> None:
        from src.prediction.prediction_pipeline import output_top_worst_results, predict_all_unified

        output_rows = predict_all_unified()
        output_top_worst_results(
            output_rows,
            mode="unified",
            shadow_mode=True,
            model_version=self._model_version,
        )


class ModelTrainingExecutor:
    """prediction BC: 統合モデル学習パイプライン。"""

    def __init__(self, model_type: str, model_name: str) -> None:
        self._model_type = model_type
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"training_{self._model_name}"

    def execute(self) -> None:
        from src.prediction.unified_model_pipeline import train_unified_model

        train_unified_model(model_type=self._model_type, model_name=self._model_name)


class BacktestOptimizationExecutor:
    """backtest BC: バックテスト最適化パイプライン。"""

    @property
    def name(self) -> str:
        return "backtest_optimization"

    def execute(self) -> None:
        from src.watchlist.batch_runner import load_target_symbols

        use_optuna = os.getenv("USE_OPTUNA", "").strip().lower() in ("1", "true", "yes")
        n_trials = int(os.getenv("OPTUNA_N_TRIALS", "50"))
        tasks = load_target_symbols()

        if use_optuna:
            from src.backtest.optimizer import run_optuna_batch

            run_optuna_batch(
                tasks,
                model_type="XGBoostModel",
                ensemble=False,
                source="file",
                n_splits=5,
                n_trials=n_trials,
                max_workers=3,
                sort_by="sharpe_ratio",
            )
        else:
            from src.backtest.optimizer import run_optimize_batch

            run_optimize_batch(
                tasks,
                model_type="XGBoostModel",
                ensemble=False,
                source="file",
                n_splits=5,
                max_workers=3,
                sort_by="sharpe_ratio",
            )


class WatchlistRefreshExecutor:
    """watchlist BC: ウォッチリスト自動更新パイプライン。"""

    @property
    def name(self) -> str:
        return "watchlist_refresh"

    def execute(self) -> None:
        from src.watchlist.manager import run_watchlist_refresh

        run_watchlist_refresh()


class PipelineOrchestrator:
    """
    Executor リストを受け取るスケジューラー DI コンテナ。

    critical_executors が失敗すると例外を再送出し後続処理を停止する。
    noncritical_executors が失敗してもログを記録して後続に進む。

    Usage::

        orchestrator = PipelineOrchestrator(
            critical_executors=[DataPipelineExecutor(), PredictionExecutor()],
            noncritical_executors=[WatchlistRefreshExecutor()],
        )
        orchestrator.run()
    """

    def __init__(
        self,
        critical_executors: list[PipelineExecutor] | None = None,
        noncritical_executors: list[PipelineExecutor] | None = None,
    ) -> None:
        self._critical = list(critical_executors or [])
        self._noncritical = list(noncritical_executors or [])

    def run(self) -> None:
        for executor in self._critical:
            logger.info("[PipelineOrchestrator] 開始: %s", executor.name)
            try:
                executor.execute()
                logger.info("[PipelineOrchestrator] 完了: %s", executor.name)
            except Exception as e:
                logger.error(
                    "[PipelineOrchestrator] 致命的エラー (%s): %s",
                    executor.name,
                    e,
                    exc_info=True,
                )
                raise

        for executor in self._noncritical:
            logger.info("[PipelineOrchestrator] 開始: %s", executor.name)
            try:
                executor.execute()
                logger.info("[PipelineOrchestrator] 完了: %s", executor.name)
            except Exception as e:
                logger.error(
                    "[PipelineOrchestrator] エラー（継続）(%s): %s",
                    executor.name,
                    e,
                    exc_info=True,
                )
