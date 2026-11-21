"""日次の [3.5/6] Challenger shadow 予測の段（#1139）。

例外で予想を出せなかった銘柄は、影の比べの的中率を黙って良く見せうるため、
後の段は止めずに件数と先頭の銘柄を1回知らせる。challenger のファイルが在るのに
予想が0件の日は、モデルが無い日と分けてログに出す。
"""

from typing import Sequence

from src.orchestration.jobs.common import _handle_stage_error, _notify_stage_failure
from src.orchestration.jobs.prediction_errors import notify_prediction_errors
from src.orchestration.types import PipelineStage
from src.prediction.types import PredictionResult
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
            _log_same_direction(challenger_rows)
        elif errors or available_unified_challengers():
            logger.warning("%s: Challenger モデルは在るが予想が0件だった", _STAGE)
        else:
            logger.info("[3.5/6] Challenger モデルなし（スキップ）")
        notify_prediction_errors(errors, stage=_STAGE)
    except Exception as e:
        # 段全体が落ちた日は影の予想が丸ごと欠けるため、後の段は止めずに知らせる（#1368）
        _handle_stage_error(PipelineStage.NON_CRITICAL, _STAGE, e)
        _notify_stage_failure(_STAGE, e)


def _log_same_direction(challenger_rows: Sequence[PredictionResult]) -> None:
    """production の A-4 と同じ線で、市場ごとの向きの偏りをログに出す（#2080）。

    見える化だけで、予想・保存・知らせは変えない。判定が落ちても段は止めない。
    """
    try:
        from src.prediction.output_invariants import check_same_direction

        by_market: dict[str, list[PredictionResult]] = {}
        for r in challenger_rows:
            by_market.setdefault(r.market, []).append(r)

        biased = False
        summaries: list[str] = []
        for market in sorted(by_market):
            rows = by_market[market]
            up = sum(1 for r in rows if r.diff_ratio > 0)
            down = sum(1 for r in rows if r.diff_ratio < 0)
            total = len(rows)
            summaries.append(f"{market} 上げ {up}・下げ {down}・総数 {total}")
            if check_same_direction(rows):
                biased = True
                majority, label = (up, "上げ") if up >= down else (down, "下げ")
                logger.warning(
                    "[3.5/6] Challenger の予想が同じ向きに偏った: %s %d/%d 銘柄が%s",
                    market,
                    majority,
                    total,
                    label,
                )
        if not biased:
            logger.info("[3.5/6] Challenger の予想の向き: %s", "、".join(summaries))
    except Exception as e:
        logger.warning("[3.5/6] Challenger の予想の同じ向きの判定に失敗: %s", e, exc_info=True)
