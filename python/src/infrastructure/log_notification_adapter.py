"""ログ出力通知アダプター（開発環境用）

NotificationPort ポートのログ出力実装。
Discord Webhook なしでアラートをローカルログに記録する。
"""

from typing import Optional

from src.domain.ports import AlertLevel, NotificationPort
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LogNotificationAdapter(NotificationPort):
    """ログ出力のみ行う通知アダプター（開発環境・デバッグ用）"""

    def send_message(self, message: str, webhook_url: Optional[str] = None) -> bool:
        logger.info("[通知] %s", message)
        return True

    def send_alert(self, title: str, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        if level == AlertLevel.ERROR:
            logger.error("[%s] %s: %s", level.name, title, message)
        elif level == AlertLevel.WARNING:
            logger.warning("[%s] %s: %s", level.name, title, message)
        else:
            logger.info("[%s] %s: %s", level.name, title, message)
        return True
