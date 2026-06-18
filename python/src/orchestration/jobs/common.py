"""スケジューラジョブ共通ヘルパー。

各 cadence 別ジョブモジュール（daily / weekly / periodic）から共有される
エラーハンドリングと日付判定ユーティリティを提供する。
"""

from typing import TYPE_CHECKING, Callable, Optional

from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from datetime import datetime

logger = get_logger(__name__)


def _handle_stage_error(
    stage: PipelineStage,
    label: str,
    exc: Exception,
    notify_fn: Optional[Callable[[str], object]] = None,
) -> bool:
    """ステージ分類に基づくエラーハンドリング。

    Returns:
        True  → 呼び出し元は raise すべき (CRITICAL)
        False → 継続してよい (NON_CRITICAL / RECOVERABLE)
    """
    if stage is PipelineStage.CRITICAL:
        logger.error("%s 失敗: %s", label, exc, exc_info=True)
        if notify_fn is not None:
            notify_fn(f"{label} 失敗: {exc}")
        return True
    elif stage is PipelineStage.NON_CRITICAL:
        logger.error("%s 失敗: %s", label, exc, exc_info=True)
        return False
    else:  # RECOVERABLE
        logger.warning("%s 失敗（継続）: %s", label, exc, exc_info=True)
        return False


def _is_first_week_of_month(now: "datetime") -> bool:
    """月初週（1〜7日）かどうか。月次オートコンパクションの発火判定に使う。

    週次メンテは土曜に走るため、各月で 1〜7 日に当たる土曜は必ず1回だけ存在する。
    これにより「月1回」のコンパクションを単純な日付判定で実現する。
    """
    return now.day <= 7
