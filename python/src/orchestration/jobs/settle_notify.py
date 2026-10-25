"""ペーパートレードの約定の知らせ（#1088 で daily の run_daily_settle_orders から分けた）

完了の知らせとは別に、例外で約定できなかった注文の件数を、当日の行が無い据え置きと分けて知らせる。
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def notify_settle_result(settled_count: int, error_count: int, notify_when_none: bool) -> int:
    """約定の完了と例外の件数を知らせ、例外の件数を返す"""
    try:
        from src.reporting.discord.discord_utils import send_daily_settle_completion

        if settled_count or notify_when_none:  # Discord 完了通知。やり直しの0件の回は出さない #1047
            send_daily_settle_completion(settled_count=settled_count)
    except Exception as e:
        logger.error("約定処理完了通知失敗: %s", e, exc_info=True)

    if error_count <= 0:
        return 0
    message = (
        f"約定処理の例外で約定できなかった注文が {error_count} 件ある。"
        "書きかけは戻して pending のまま据え置いた（当日の行が無い据え置きとは別）"
    )
    try:
        from src.reporting.discord.discord_utils import send_webhook_notification

        send_webhook_notification("❌ ペーパートレード約定処理エラー", message, color=0xFF0000)
    except Exception as e:
        logger.error("約定処理エラーの通知失敗: %s", e, exc_info=True)
    return error_count
