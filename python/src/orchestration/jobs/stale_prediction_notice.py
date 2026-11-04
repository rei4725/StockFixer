"""自動発注で、今日の予想が無いため新規を止めた日の知らせ（#1394）。"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def notify_stale_prediction(stats) -> None:
    """今日の予想が無く新規を止めた日に、件数と理由を1回知らせる（#1394）。

    予想の行が0件の日は、予想の売りも判定していないことを理由の文が伝える（#1572）。
    """
    reason = stats.get("stale_prediction_reason")
    if not reason:
        return
    try:
        from src.domain.ports import AlertLevel
        from src.infrastructure.discord_notification_adapter import DiscordNotificationAdapter

        DiscordNotificationAdapter().send_alert(
            "自動発注: 今日の予想が無いため新規を止めた",
            f"{reason}\n新規の買い・ショートは出さず、決済だけ行った"
            f"（売り={stats.get('sell_orders', 0)} 件、スキップ={stats.get('skipped', 0)} 件）。",
            level=AlertLevel.WARNING,
        )
    except Exception as e:
        logger.error("新規を止めた知らせの送信失敗: %s", e, exc_info=True)
