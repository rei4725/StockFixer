"""日次の [3.5/6] Challenger shadow 予測の段（#1139）。

例外で予想を出せなかった銘柄は、影の比べの的中率を黙って良く見せうるため、
後の段は止めずに件数と先頭の銘柄を1回知らせる。challenger のファイルが在るのに
予想が0件の日は、モデルが無い日と分けてログに出す。
"""

from src.orchestration.jobs.common import _handle_stage_error
from src.orchestration.jobs.prediction_errors import notify_prediction_errors
from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

logger = get_logger(__name__)

_STAGE = "[3.5/6] Challenger shadow 予測"


def run_challenger_shadow_stage() -> None:
    """challenger の影の予想を出して保存する（NON_CRITICAL: 失敗しても後の段へ進む）。"""
    logger.info("%s開始", _STAGE)
    try:
        from src.prediction.prediction_pipeline import output_top_worst_results
        from src.prediction.shadow_evaluation import (
            available_unified_challengers,
            predict_with_challenger_unified,
        )

        errors: list[tuple[str, str]] = []
        challenger_rows = predict_with_challenger_unified(
            on_error=lambda m, s: errors.append((m, s))
        )
        if challenger_rows:
            output_top_worst_results(
                challenger_rows, mode="unified", shadow_mode=True, model_version="challenger"
            )
            logger.info("%s完了: %d 銘柄", _STAGE, len(challenger_rows))
        elif errors or available_unified_challengers():
            logger.warning("%s: Challenger モデルは在るが予想が0件だった", _STAGE)
        else:
            logger.info("[3.5/6] Challenger モデルなし（スキップ）")
        notify_prediction_errors(errors, stage=_STAGE)
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, _STAGE, e)
