"""発注の回ごとに、新規を止めた判定を JST の日付の JSON に書く（8:50 は #1511、16:30 は #1514）。

置き場: 8:50 は results/orders/<JST の日付>.json、16:30 のルールの道は
results/orders/rule/<JST の日付>.json（8:50 の記録を上書きしない。同じ道のやり直しは上書きする）。
ゲートの判定はそのまま写す。書き込みの失敗は ERROR のログで知らせ、発注は止めない。
"""

import json
from dataclasses import asdict, dataclass, field
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


@dataclass
class RuleGateRecord:
    """16:30 のルールの道で新規買いを止めた判定（#1514）。

    risk_gate_allowed・correlation_gate_allowed は判定しなかったとき None。
    skip_reasons は skipped に数えた銘柄の理由ごとの数で、シグナルの無い銘柄は no_signal に分ける。
    """

    market: str
    allow_new_buys: bool
    risk_gate_allowed: Optional[bool] = None
    risk_gate_reason: Optional[str] = None
    risk_gate_reason_code: Optional[str] = None
    correlation_gate_allowed: Optional[bool] = None
    correlation_gate_reason: Optional[str] = None
    enc: Optional[float] = None
    new_buy_step_error: Optional[str] = None
    skip_reasons: dict[str, int] = field(default_factory=dict)
    buy_orders: int = 0
    sell_orders: int = 0
    skipped: int = 0
    errors: int = 0


def save_rule_gate_record(record: RuleGateRecord) -> Optional[str]:
    """ルールの道の判定を results/orders/rule/ に書き、パスを返す。書けなければ None を返す。"""
    try:
        now = now_jst()
        day = now.date().isoformat()
        dest = Path(get_results_dir()) / "orders" / "rule" / f"{day}.json"
        payload: dict[str, Any] = {"date": day, "written_at": now.isoformat(), **asdict(record)}
        _write_record(dest, payload)
    except Exception:
        logger.error("[rule] 新規を止めた判定の記録に失敗（発注は続ける）", exc_info=True)
        return None
    logger.info("[rule] 新規を止めた判定の記録を保存: %s", dest)
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
