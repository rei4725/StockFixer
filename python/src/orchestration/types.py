"""
orchestration BC の型定義。

スケジューラ・ジョブ管理で使用するドメイン型。
"""

from __future__ import annotations

from enum import Enum

from src.utils.scheduler_types import SchedulerJobStatus  # noqa: F401  # re-export

__all__ = ["PipelineStage", "SchedulerJobStatus"]


class PipelineStage(Enum):
    """パイプライン段階の重要度分類。

    CRITICAL:     失敗時にパイプラインを停止し Discord 通知する。
    NON_CRITICAL: 失敗してもパイプラインを継続し、エラーログのみ記録する。
    RECOVERABLE:  失敗しても継続し、警告ログを記録する（一時的な障害を想定）。
    """

    CRITICAL = "CRITICAL"
    NON_CRITICAL = "NON_CRITICAL"
    RECOVERABLE = "RECOVERABLE"
