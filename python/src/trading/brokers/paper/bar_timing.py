"""ペーパーの約定で、注文と足の日付・時刻を JST で比べる関数（paper_broker から分けた）"""

from datetime import date, datetime, time
from typing import Any

import pandas as pd

from src.domain.market_close import MARKET_CLOSE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 東証のタイムゾーンと引け（JST）。表から読む（#2121）。
# 引けより後に出た注文は、その日の足で約定させない（#1312）
_JST: str = MARKET_CLOSE["jp"][0].key
_JP_CLOSE: time = MARKET_CLOSE["jp"][1]


def parse_predicted_date(predicted_at: Any) -> date | None:
    """predicted_at（例: "20260213_142903"）からシグナルを作った日を返す。読めなければ None"""
    if not predicted_at:
        return None
    text = str(predicted_at)
    try:
        return datetime.strptime(text[:8], "%Y%m%d").date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        logger.warning(f"[paper] predicted_at を日付として読めません: {text}")
        return None


def bar_date_jst(index_value: Any) -> date:
    """OHLCV の行の日付を JST の日付で返す（tz なしの index はそのまま日付とみなす）"""
    ts = pd.Timestamp(index_value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(_JST)
    return ts.date()


def created_after_bar_close(created_at: Any, bar_date: date) -> bool:
    """注文が足の日の東証の引け（15:30 JST）以後に出たか（#1312）

    created_at の列は tz 無しの timestamp で、DB のセッションは UTC のため、tz の無い値は UTC とみなす。
    足の日より後の日に出た注文は、足が今日の行に限られるため本番では起きず、見ない。
    """
    if created_at is None:
        return False
    ts = pd.Timestamp(created_at)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    created_jst = ts.tz_convert(_JST)
    return created_jst.date() == bar_date and created_jst.time() >= _JP_CLOSE


def _created_date_jst(created_at: Any) -> date:
    ts = pd.Timestamp(created_at)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(_JST).date()


def bar_closed(bar_date: date, now: datetime) -> bool:
    """足の日の東証の引け（15:30 JST）を now（tz つき）が過ぎ、足が1日の値動きを持つか（#2108）"""
    close_at = pd.Timestamp(datetime.combine(bar_date, _JP_CLOSE)).tz_localize(_JST)
    return pd.Timestamp(now) >= close_at


def held_limit_bar_position(
    hist: pd.DataFrame,
    signal_date: date | None,
    created_at: Any,
    today: date,
    now: datetime,
) -> int | None:
    """場中の足で据え置いた指値を判定する、今日より前の引けた足の位置（#2108）

    注文が有効な日（シグナルの日より後・注文を出した日以後・注文が引けの後に出ていない）の
    うち最も早い足が、今日より前で引けていればその位置を返す。無ければ None（今日の行で判定する）。
    """
    if created_at is None:
        return None
    created_date = _created_date_jst(created_at)
    for pos in range(len(hist)):
        day = bar_date_jst(hist.index[pos])
        if signal_date is not None and day <= signal_date:
            continue
        if day < created_date or created_after_bar_close(created_at, day):
            continue
        if day < today and bar_closed(day, now):
            return pos
        return None
    return None
