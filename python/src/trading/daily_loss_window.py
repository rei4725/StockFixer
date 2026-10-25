"""日次損失を数える時刻の幅（#941）"""

from datetime import datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from src.domain.jp_market_holidays import is_jp_business_day

_JST = ZoneInfo("Asia/Tokyo")


def _utc_now() -> datetime:
    """今の時刻（UTC）を返す。テストで差し替える。"""
    return datetime.now(timezone.utc)


def daily_loss_window_start(now_utc: Optional[datetime] = None) -> datetime:
    """日次損失を数え始める時刻（前の営業日の 0:00 JST）を naive の UTC で返す。

    8:50 JST の朝の判定では JST の当日の約定がまだ無いので、前の営業日（月曜は金曜）
    の確定損失を数える。DB の CURRENT_DATE は UTC の日付で、月曜の朝は日曜を見て
    金曜の損失を数えなかった。DB の時刻は UTC で入っているので naive の UTC で比べる。
    営業日は土日と東証の休場日（祝日・年末年始）を除く。祝日の月曜の次の火曜は金曜を
    数える（#1050）。休場日の表に無い年は土日だけを除く。
    """
    if now_utc is None:
        now_utc = _utc_now()
    prev_day = now_utc.astimezone(_JST).date() - timedelta(days=1)
    while not is_jp_business_day(prev_day):
        prev_day -= timedelta(days=1)
    start_jst = datetime.combine(prev_day, time.min, tzinfo=_JST)
    return start_jst.astimezone(timezone.utc).replace(tzinfo=None)
