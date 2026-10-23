"""運用アラートの連続回数の更新（NF-303-2・3、#934）。

日次の道（自動発注・ドリフトチェック）の結果から呼ぶ。更新の失敗で日次の道を止めない。
daily.py の行数ガード対策で、呼び出し側は1行で済むようにここへ置く。
"""

from src.utils.alert_service import update_drift_warn_streak, update_loss_limit_streak
from src.utils.logger import get_logger

logger = get_logger(__name__)


def record_loss_limit_streak(stats: dict) -> None:
    """自動発注の結果から損失上限（R-003）の連続回数を更新する。

    trading_stopped は連続損失やドローダウンの停止でも真になるため、
    損失上限による停止（reason_code == "daily_loss_limit"）だけを数える。
    """
    try:
        update_loss_limit_streak(stats.get("reason_code") == "daily_loss_limit")
    except Exception as e:
        logger.error("損失上限ストリーク更新失敗: %s", e, exc_info=True)


def record_drift_warn_streak(warn_detected: bool) -> None:
    """日次ドリフトチェックの判定からドリフト警告の連続回数を更新する。"""
    try:
        update_drift_warn_streak(warn_detected)
    except Exception as e:
        logger.error("ドリフト警告ストリーク更新失敗: %s", e, exc_info=True)
