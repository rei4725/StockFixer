"""発注の回ごとに、新規を止めた判定を JST の日付の JSON に書く（8:50 は #1511、16:30 は #1514）。

置き場: 8:50 は results/orders/<JST の日付>.json、16:30 のルールの道は
results/orders/rule/<JST の日付>.json（8:50 の記録を上書きしない。同じ道のやり直しは上書きする）。
ゲートの判定はそのまま写す。8:50 の記録には SL/TP を評価できなかった保有の銘柄と数も写す（#1516）。
売りの判定に使える予想が無かった保有の銘柄と数も写し、判定をしなかった日は null にする（#1530）。
16:30 の記録にはシグナルを出せなかった銘柄と、そのうち保有中の銘柄の一覧と数も写す（#1520）。
その日のルールの判定の対象に入らなかった保有の一覧と数も写す（#1527）。
書き込みの失敗は ERROR のログで知らせ、発注は止めない。
"""

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import pandas as pd

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


def held_without_prediction(predictions: pd.DataFrame, held_symbols: Collection[str]) -> list[str]:
    """統合スコアのある予想の行が無い保有を銘柄の順に返し、WARNING に出す（#1530）。

    行の無い保有と統合スコアが NaN の保有は、売りシグナルの比べが成り立たない。
    """
    scored = predictions.loc[predictions["multi_horizon_score"].notna(), "symbol"]
    missing = sorted(set(held_symbols) - set(scored))
    if missing:
        logger.warning(
            "[exec] 予想の行が無いか統合スコアが NaN で売りの判定ができない保有が %d件: %s",
            len(missing),
            missing,
        )
    return missing


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
        # SL/TP を評価できずに飛ばした保有（#1516）。判定はせず stats を写すだけ
        unevaluable = list(stats.get("sl_tp_unevaluable_symbols") or [])
        payload["sl_tp_unevaluable_symbols"] = unevaluable
        payload["sl_tp_unevaluable_count"] = len(unevaluable)
        no_prediction = stats.get("held_without_prediction_symbols")
        payload["held_without_prediction_symbols"] = no_prediction
        payload["held_without_prediction_count"] = (
            None if no_prediction is None else len(no_prediction)
        )
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


class UnavailableSymbol(Protocol):
    """シグナルを出せなかった銘柄（rule_engine の UnavailableRuleSymbol の形。#1520）。

    BC どうしは import しないので、記録に要る属性だけをここに書く。
    """

    symbol: str
    rule: str
    reason: str


def save_rule_gate_record(
    record: RuleGateRecord,
    *,
    unavailable_symbols: Sequence[UnavailableSymbol] = (),
    held_symbols: Collection[str] = (),
    signals: Sequence[Mapping[str, Any]] = (),
) -> Optional[str]:
    """ルールの道の判定を results/orders/rule/ に書き、パスを返す。書けなければ None を返す。

    unavailable_symbols はシグナルを出せなかった銘柄で、held_symbols（約定済みの保有）に
    入るものは、その日のルールの売りを判定できなかった保有として別の一覧にも写す（#1520）。
    held_symbols のうち signals にも unavailable_symbols にも入らない銘柄は、その日のルールの
    判定の対象に入らなかった保有として、銘柄の順の一覧と数に写す（#1527）。
    """
    try:
        now = now_jst()
        day = now.date().isoformat()
        dest = Path(get_results_dir()) / "orders" / "rule" / f"{day}.json"
        payload: dict[str, Any] = {"date": day, "written_at": now.isoformat(), **asdict(record)}
        rows = [
            {"symbol": u.symbol, "rule": u.rule, "reason": u.reason} for u in unavailable_symbols
        ]
        held_rows = [row for row in rows if row["symbol"] in held_symbols]
        payload["unavailable_symbols"] = rows
        payload["unavailable_count"] = len(rows)
        payload["unavailable_held_symbols"] = held_rows
        payload["unavailable_held_count"] = len(held_rows)
        judged = {item["symbol"] for item in signals} | {row["symbol"] for row in rows}
        unjudged = sorted(set(held_symbols) - judged)
        payload["unjudged_held_symbols"] = unjudged
        payload["unjudged_held_count"] = len(unjudged)
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
