"""Claude の道で新規の buy を止める日の判定（#1227・#1490）。

8:50 の道と同じ規則で、止めるときは理由を返す。保有の sell は止めない。
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from src.trading.execution.freshness import _block_new_if_stale
from src.trading.execution.params import _PARAMS_UNREADABLE_COL
from src.trading.execution.stats import OrderExecutionStats
from src.trading.types import TradingGateStatus

# optimal_params.json が読めない日に新規の buy を止める理由（#1227、8:50 の道の #1214 と同じ止め方）
_PARAMS_UNREADABLE_REASON = (
    "optimal_params.json が読めないため新規の buy を止める（保有の sell は可）"
)


def _params_unreadable(predictions: pd.DataFrame) -> bool:
    """閾値を既定値で決めた行があるか。あれば新規の buy を止める（#1227）。"""
    if _PARAMS_UNREADABLE_COL not in predictions.columns:
        return False
    return bool(predictions[_PARAMS_UNREADABLE_COL].astype(bool).any())


def _stale_prediction_reason(
    predictions: pd.DataFrame, saved_day: str | None, stats: dict[str, Any]
) -> str | None:
    """最新の回と保存の終わりの印が今日（JST）でなければ、新規の buy を止める理由を返す（#1490）。

    規則は 8:50 の道の _block_new_if_stale（#1394）と同じ。saved_day が None なら確かめない。
    止めるときは stats["stale_prediction_reason"] に理由が入る。
    """
    if "predicted_at" not in predictions.columns:
        predictions = pd.DataFrame({"predicted_at": pd.Series(dtype=str)})
    gate = _block_new_if_stale(
        predictions,
        TradingGateStatus(is_allowed=True, stop_active=False),
        saved_day,
        cast(OrderExecutionStats, stats),
    )
    if gate.is_allowed:
        return None
    return f"{gate.reason}ため新規の buy を止める（保有の sell は可）"
