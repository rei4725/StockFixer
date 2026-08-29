"""スケジューラジョブ共通ヘルパー。

各 cadence 別ジョブモジュール（daily / weekly / periodic）から共有される
エラーハンドリングと日付判定ユーティリティを提供する。
"""

import threading
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional

from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from datetime import datetime

logger = get_logger(__name__)

# run_daily_drift_check() 専用の多重起動防止ロック（daily.py / skip_if_running から共有）。
_drift_check_lock = threading.Lock()


def skip_if_running(lock: threading.Lock, job_label: str) -> Callable:
    """同一プロセス内で多重起動された場合、後続の呼び出しは実処理をスキップするデコレータ。

    複数の呼び出し経路（cronジョブ・日次パイプライン内・手動実行）を持つ
    重い処理が並行実行されるのを防ぐための最終防衛ライン。
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            if not lock.acquire(blocking=False):
                logger.warning("%s: 既に実行中のためスキップ（多重起動防止）", job_label)
                return None
            try:
                return func(*args, **kwargs)
            finally:
                lock.release()

        return wrapper

    return decorator


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
