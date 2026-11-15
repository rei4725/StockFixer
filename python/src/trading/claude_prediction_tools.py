"""
ClaudeTrader の tool `get_predictions` の手当て。

claude_agent.py から分けたもの（#1849）。動きは分ける前と同じ。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.domain.ports import PredictionResultRepository
from src.trading.brokers.base import BrokerBase
from src.trading.claude_buy_blocks import _PARAMS_UNREADABLE_REASON, _params_unreadable
from src.trading.execution import _attach_dynamic_thresholds, _get_held_symbols
from src.trading.signal_generator import apply_multi_horizon_score_column
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Claude に渡す予測候補の上限行数（トークン節約）
_MAX_PREDICTION_ROWS = 50


def _handle_get_predictions(
    market: str,
    broker: BrokerBase,
    prediction_repo: PredictionResultRepository,
    buy_block_reason: str | None = None,
) -> dict[str, Any]:
    """閾値通過候補の予測を返す。トークン節約のため上位 N 行に絞る。

    buy_block_reason があれば（予想が古い日など、#1490）buy の候補を空にする。
    """
    try:
        predictions = prediction_repo.get_latest_by_market(market)
        if predictions.empty:
            return {"candidates": [], "message": "予測結果なし"}

        predictions = _attach_dynamic_thresholds(predictions)
        predictions = apply_multi_horizon_score_column(predictions)

        held_symbols = _get_held_symbols(broker)
        buy_blocked = buy_block_reason is not None or _params_unreadable(predictions)

        buy_candidates = predictions[
            (~predictions["symbol"].isin(held_symbols))
            & (predictions["multi_horizon_score"] >= predictions["effective_buy_threshold"])
        ].head(_MAX_PREDICTION_ROWS)
        if buy_blocked:
            buy_candidates = buy_candidates.iloc[0:0]

        sell_candidates = predictions[
            (predictions["symbol"].isin(held_symbols))
            & (predictions["multi_horizon_score"] <= predictions["effective_sell_threshold"])
        ]

        def _to_records(df: pd.DataFrame) -> list[dict]:
            cols = [
                "market",
                "symbol",
                "current_price",
                "diff_ratio",
                "diff_ratio_3d",
                "diff_ratio_5d",
                "diff_ratio_10d",
                "confluence_score",
                "confidence_ratio",
                "multi_horizon_score",
                "effective_buy_threshold",
                "effective_sell_threshold",
            ]
            available = [c for c in cols if c in df.columns]
            return df[available].to_dict(orient="records")

        result: dict[str, Any] = {
            "buy_candidates": _to_records(buy_candidates),
            "sell_candidates": _to_records(sell_candidates),
            "held_symbols": sorted(held_symbols),
            "threshold_scale": (
                float(predictions["threshold_scale"].iloc[0]) if not predictions.empty else 1.0
            ),
        }
        if buy_blocked:
            result["buy_blocked_reason"] = buy_block_reason or _PARAMS_UNREADABLE_REASON
        return result
    except Exception as e:
        logger.error("[claude_agent] get_predictions エラー: %s", e, exc_info=True)
        return {"error": str(e)}
