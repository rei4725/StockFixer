"""日次の [2/6] の、足が直った後の採点の付け直しの段（#2155）。

daily.py からの責務分離（ファイル行数ガード対策）。失敗は投げず、モデルごとに知らせるだけにする。
"""

from src.orchestration.jobs.common import _handle_stage_error, _notify_stage_failure
from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

logger = get_logger(__name__)


def rescore_accuracy_stage() -> None:
    """production と challenger の直近の採点の行を、直った足の終値で付け直す（NON_CRITICAL）。"""
    for model_name in ("production", "challenger"):
        label = f"[2/6] 採点の付け直し ({model_name})"
        try:
            from src.prediction.accuracy_rescore import rescore_fixed_bars

            rescore_fixed_bars(model_name=model_name)
        except Exception as e:
            _handle_stage_error(PipelineStage.NON_CRITICAL, label, e)
            _notify_stage_failure(label, e)
