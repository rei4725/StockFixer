"""紙以外のブローカーの実現損益を trade_pnl から読む（#995）

数える元（trade_pnl）が無い・読めないときは 0 でなく None（分からない）を返す。
ゲートは None を受けたら、理由のコードを分けて新規の買いを止める側に倒す。
"""

from datetime import datetime
from typing import Any, Optional

from config.settings import MAX_CONSECUTIVE_LOSSES, MAX_POSITIONS
from src.trading.consecutive_losses import count_leading_losses, recent_trade_pnl_closed_pnls
from src.trading.types import TradingGateStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)

DAILY_LOSS_SOURCE_UNAVAILABLE = "daily_loss_source_unavailable"
CONSECUTIVE_LOSS_SOURCE_UNAVAILABLE = "consecutive_loss_source_unavailable"


def table_exists(con: Any, table_name: str) -> bool:
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = %s
        """,
        [table_name],
    ).fetchone()
    return bool(row and row[0])


def _trade_pnl_ready(con: Any, broker_name: str, what: str) -> bool:
    if table_exists(con, "trade_pnl"):
        return True
    logger.warning(
        f"[risk] trade_pnl テーブル未作成のため{what}を数えられません。新規の買いを止めます: "
        f"broker={broker_name}"
    )
    return False


def trade_pnl_daily_loss(con: Any, broker_name: str, window_start: datetime) -> Optional[float]:
    """区切りからの確定損失合計（プラスが損失）。数える元が無い・読めないときは None"""
    try:
        if not _trade_pnl_ready(con, broker_name, "日次損失"):
            return None
        row = con.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0.0)
            FROM trade_pnl
            WHERE closed_at >= %s
              AND broker = %s
              AND realized_pnl < 0
            """,
            [window_start, broker_name],
        ).fetchone()
    except Exception:
        logger.error(
            f"[risk] trade_pnl を読めないため日次損失を数えられません: broker={broker_name}",
            exc_info=True,
        )
        return None
    return abs(float(row[0])) if row else 0.0


def trade_pnl_consecutive_losses(
    con: Any, broker_name: str, window_start: datetime, limit: int
) -> Optional[int]:
    """区切りからの決済の直近の連続損失回数。数える元が無い・読めないときは None"""
    try:
        if not _trade_pnl_ready(con, broker_name, "連続損失"):
            return None
        rows = recent_trade_pnl_closed_pnls(con, broker_name, window_start, limit)
    except Exception:
        logger.error(
            f"[risk] trade_pnl を読めないため連続損失を数えられません: broker={broker_name}",
            exc_info=True,
        )
        return None
    return count_leading_losses(rows)


def source_unavailable_status(
    reason_code: str,
    daily_loss: float,
    daily_loss_limit: Optional[float],
    consecutive: int,
    n_positions: int,
) -> TradingGateStatus:
    """数える元が無い・読めないときの不合格のゲート（新規の買いを止める）"""
    reason = f"実現損益の数える元が無い・読めない: {reason_code}"
    logger.warning(f"[risk] {reason} → 新規発注停止")
    return TradingGateStatus(
        is_allowed=False,
        stop_active=True,
        reason_code=reason_code,
        reason=reason,
        daily_loss=daily_loss,
        daily_loss_limit=daily_loss_limit,
        consecutive_losses=consecutive,
        consecutive_loss_limit=MAX_CONSECUTIVE_LOSSES,
        position_count=n_positions,
        max_positions=MAX_POSITIONS,
    )
