"""約定のやり直しのジョブ（#1008）。

9:05 の約定で当日の行が無く pending に据え置いた注文を、同じ朝のうちに当日始値で約定し直す。
約定の処理そのものと、それを動かすかの判定は daily の run_daily_settle_orders に任せる。
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.orchestration.jobs.daily import run_daily_settle_orders
from src.utils.logger import get_logger

logger = get_logger(__name__)

_JST = ZoneInfo("Asia/Tokyo")

# 約定のやり直しの締め（JST）。この時刻以降の回で残った注文は、翌営業日まで据え置く
SETTLE_RETRY_CUTOFF_JST = (9, 55)


def count_pending_paper_orders() -> int:
    """約定待ち（pending）のペーパートレード注文の件数を返す。"""
    from src.utils.db._connection import _db_connection

    with _db_connection() as con:
        row = con.execute("SELECT COUNT(*) FROM paper_orders WHERE status = 'pending'").fetchone()
    return int(row[0]) if row else 0


def run_daily_settle_retry(now: datetime | None = None) -> None:
    """
    毎営業日 9:15〜9:55 実行: 据え置いた pending 注文の約定をやり直す。
    約定待ちが無ければ約定の処理を動かさない。締めの回で残れば WARNING と知らせを1回出す。
    注文は取り消さない（残った注文は翌営業日まで据え置く）。
    """
    now_jst = (now or datetime.now(_JST)).astimezone(_JST)
    is_final = (now_jst.hour, now_jst.minute) >= SETTLE_RETRY_CUTOFF_JST

    pending_count = count_pending_paper_orders()
    if pending_count == 0:
        logger.info("約定待ちの注文が無いため約定のやり直しをスキップ")
        return

    logger.info("=== 約定のやり直し: 約定待ち %d 件 ===", pending_count)
    # 約定が0件の回は完了の知らせを出さない（9:05 の分と重ねない。#1047）
    run_daily_settle_orders(notify_when_none=False)

    if not is_final:
        return

    remaining = count_pending_paper_orders()
    if remaining <= 0:
        return

    message = (
        f"締め {SETTLE_RETRY_CUTOFF_JST[0]}:{SETTLE_RETRY_CUTOFF_JST[1]:02d} を過ぎても"
        f"当日の行が無く約定待ちの注文が {remaining} 件残った。翌営業日まで据え置く"
    )
    logger.warning("[paper] %s", message)
    try:
        from src.reporting.discord.discord_utils import send_webhook_notification

        send_webhook_notification("⚠️ ペーパートレード約定待ちの残り", message, color=0xFFA500)
    except Exception as e:
        logger.error("約定待ちの残りの通知失敗: %s", e, exc_info=True)
