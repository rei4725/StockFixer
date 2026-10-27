"""運用アラートの連続回数の更新（NF-303-2・3・4、#934、#940）。

日次の道（自動発注・ドリフトチェック・日次パイプライン）の結果から呼ぶ。更新の失敗で日次の道を止めない。
daily.py の行数ガード対策で、呼び出し側は1行で済むようにここへ置く。
JST の今日を渡し、同じ日のやり直しの回で二重に数えない（#1242）。
"""

from src.orchestration.jobs.common import today_jst_key
from src.utils.alert_service import (
    update_drift_warn_streak,
    update_health_degraded_streak,
    update_loss_limit_streak,
)
from src.utils.health_check import is_degraded_for_daily
from src.utils.logger import get_logger

logger = get_logger(__name__)


def record_loss_limit_streak(stats: dict) -> None:
    """自動発注の結果から損失上限（R-003）の連続回数を更新する。

    trading_stopped は連続損失やドローダウンの停止でも真になるため、
    損失上限による停止（reason_code == "daily_loss_limit"）だけを数える。
    """
    try:
        update_loss_limit_streak(stats.get("reason_code") == "daily_loss_limit", today_jst_key())
    except Exception as e:
        logger.error("損失上限ストリーク更新失敗: %s", e, exc_info=True)


def record_drift_warn_streak(warn_detected: bool) -> None:
    """日次ドリフトチェックの判定からドリフト警告の連続回数を更新する。"""
    try:
        update_drift_warn_streak(warn_detected, today_jst_key())
    except Exception as e:
        logger.error("ドリフト警告ストリーク更新失敗: %s", e, exc_info=True)


def record_health_degraded_streak() -> None:
    """日次の道で1日1回、degraded の判定から NF-303-4 の連続回数を更新する。

    /health の中では更新しない（HEALTHCHECK が短い間隔で叩き、しきい値にすぐ届くため）。
    判定はスケジューラの鮮度を見ない is_degraded_for_daily を使う。
    """
    try:
        update_health_degraded_streak(is_degraded_for_daily(), today_jst_key())
    except Exception as e:
        logger.error("ヘルス degraded ストリーク更新失敗: %s", e, exc_info=True)
