"""スケジューラ関連の共有型（utils 層から参照可能）。"""

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
