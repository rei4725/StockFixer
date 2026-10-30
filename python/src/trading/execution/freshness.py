"""今日（JST）の production の予想の保存が終わっていない朝に新規を止めるゲート（#1394）。

日次の道が遅れた朝やり直した朝の 8:50 は、最新の回が前の日の回か、今日の回の保存の
途中（書き終えた銘柄だけ）になる。そのときは新規（買い・ショート）だけ止め、決済は出す。
"""

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.trading.types import TradingGateStatus
from src.utils.logger import get_logger

from .stats import OrderExecutionStats

logger = get_logger(__name__)

_JST = ZoneInfo("Asia/Tokyo")


def _local_now() -> datetime:
    """プロセスの時計の今を、タイムゾーン付きで返す（テストで差し替える）。"""
    return datetime.now().astimezone()


def _jst_day(predicted_at: object, local_now: datetime) -> str:
    """predicted_at（YYYYMMDD_HHMMSS、プロセスの時計の時刻）の JST の日付を返す。読めなければ空。"""
    try:
        naive = datetime.strptime(str(predicted_at)[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return ""
    return naive.replace(tzinfo=local_now.tzinfo).astimezone(_JST).strftime("%Y%m%d")


def _block_new_if_stale(
    predictions: pd.DataFrame,
    gate_status: TradingGateStatus,
    saved_day: str | None,
    stats: OrderExecutionStats,
) -> TradingGateStatus:
    """最新の回と保存の終わりの印が今日（JST）でなければ、新規を止めたゲートを返す。

    saved_day は日次の道の [3/6] の保存の後に残した JST の日付（YYYYMMDD、無ければ空）。
    None なら確かめない。
    """
    if saved_day is None:
        return gate_status
    local_now = _local_now()
    today = local_now.astimezone(_JST).strftime("%Y%m%d")
    latest = str(predictions["predicted_at"].astype(str).max())
    if saved_day == today and _jst_day(latest, local_now) == today:
        return gate_status
    reason = (
        f"今日（JST {today}）の production の予想の保存が終わっていない"
        f"（最新の回={latest or '不明'} 行={len(predictions)} 件、"
        f"保存の終わりの印={saved_day or 'なし'}）"
    )
    logger.warning("[exec] %s。新規は止め、決済は続ける", reason)
    stats["stale_prediction_reason"] = reason
    return replace(gate_status, is_allowed=False, reason=gate_status.reason or reason)
