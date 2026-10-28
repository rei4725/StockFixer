"""ペーパーの約定で、注文と足の日付・時刻を JST で比べる関数（paper_broker から分けた）"""

from datetime import date, datetime, time
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_JST = "Asia/Tokyo"
# 東証の引け（JST）。これより後に出た注文は、その日の足で約定させない（#1312）
_JP_CLOSE = time(15, 30)


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
