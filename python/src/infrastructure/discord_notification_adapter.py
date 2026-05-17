"""Discord 通知アダプター

NotificationPort ポートの Discord Webhook 実装。
"""

import os
from typing import Optional

import requests

from src.domain.ports import AlertLevel, NotificationPort
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = 10


class DiscordNotificationAdapter(NotificationPort):
    """Discord Webhook を使った通知アダプター"""

    def __init__(self, default_webhook_url: Optional[str] = None) -> None:
        self._default_webhook_url = default_webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    def send_message(self, message: str, webhook_url: Optional[str] = None) -> bool:
        url = webhook_url or self._default_webhook_url
        if not url:
            logger.warning("Discord webhook URL が未設定のため通知をスキップします")
            return False
        try:
            response = requests.post(
                url,
                json={"content": message},
                timeout=_DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Discord 通知失敗: {e}", exc_info=True)
            return False

    def send_alert(self, title: str, message: str, level: AlertLevel = AlertLevel.INFO) -> bool:
        from src.reporting.discord.discord_utils import send_webhook_notification

        _level_colors = {
            AlertLevel.INFO: 0x00FF00,
            AlertLevel.WARNING: 0xFFAA00,
            AlertLevel.ERROR: 0xFF0000,
        }
        color = _level_colors.get(level, 0x00FF00)
        return send_webhook_notification(title, message, color=color)
