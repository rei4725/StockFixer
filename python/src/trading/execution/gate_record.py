"""8:50 の発注の回ごとに、新規を止めた判定を JST の日付の JSON に書く（#1511）。

置き場: results/orders/<JST の日付>.json（同じ日のやり直しは上書きする）。
ゲートの判定は stats の値をそのまま写す。書き込みの失敗は ERROR のログで知らせ、発注は止めない。
"""

import json
from pathlib import Path
from typing import Any, Optional

from src.utils.data_path_utils import get_results_dir
from src.utils.japan_time import now_jst
from src.utils.logger import get_logger

from .stats import OrderExecutionStats

logger = get_logger(__name__)

_GATE_KEYS = (
    "trading_stopped",
    "stop_reason",
    "reason_code",
    "daily_loss",
    "daily_loss_limit",
    "correlation_blocked",
    "enc",
    "avg_correlation",
    "stale_prediction_reason",
    "threshold_scale",
    "buy_orders",
    "sell_orders",
    "short_orders",
    "skipped",
    "skipped_min_change",
    "errors",
)


def _write_record(dest: Path, payload: dict[str, Any]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_order_gate_record(
    stats: OrderExecutionStats,
    *,
    market: str,
    mode: str,
    entry_error: Optional[Exception] = None,
) -> Optional[str]:
    """stats のゲートの判定を書き、ファイルのパスを返す。書けなければ None を返す。"""
    try:
        now = now_jst()
        day = now.date().isoformat()
        dest = Path(get_results_dir()) / "orders" / f"{day}.json"
        payload: dict[str, Any] = {
            "date": day,
            "written_at": now.isoformat(),
            "market": market,
            "mode": mode,
            "entry_error": repr(entry_error) if entry_error is not None else None,
        }
        for key in _GATE_KEYS:
            payload[key] = stats.get(key)
        _write_record(dest, payload)
    except Exception:
        logger.error("[exec] 新規を止めた判定の記録に失敗（発注は続ける）", exc_info=True)
        return None
    logger.info("[exec] 新規を止めた判定の記録を保存: %s", dest)
    return str(dest)


def finish_order_run(
    stats: OrderExecutionStats,
    *,
    market: str,
    mode: str,
    entry_error: Optional[Exception] = None,
) -> OrderExecutionStats:
    """判定を記録してから、新規の手順の例外があれば投げ直し、無ければ stats を返す。"""
    save_order_gate_record(stats, market=market, mode=mode, entry_error=entry_error)
    if entry_error is not None:
        raise entry_error
    return stats
