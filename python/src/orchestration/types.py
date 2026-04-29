"""
orchestration BC の型定義。

スケジューラ・ジョブ管理で使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SchedulerJobStatus:
    """スケジューラ状態表示用のジョブステータス。"""

    job_id: str
    label: str
    last_run_at: Optional[str]
    status: str
