"""新規の株数と変動の小さいスキップ（runner.py の売り・新規買い・新規ショートで同じ判断、#1865）"""

import pandas as pd

from config.settings import MIN_CHANGE_RATIO
from src.trading.position_size_reason import zero_qty_skip_key
from src.trading.risk_manager import RiskManager
from src.utils.logger import get_logger

from .params import _apply_split_qty, _resolve_kelly_params
from .stats import OrderExecutionStats, count_skip

# 発注の手順の一部なので、ログは前と同じ runner の名前で出す（テストと本番のログの name を変えない）
logger = get_logger("src.trading.execution.runner")


def _skip_small_change(
    row: pd.Series, symbol: str, stats: OrderExecutionStats, key: str, word: str
) -> bool:
    """変動が小さい（R-214）なら key で数えて True を返す。word はログの「→」の後の語。"""
    if not abs(float(row.get("diff_ratio") or 0.0)) < MIN_CHANGE_RATIO:
        return False
    logger.info(
        "[exec] %s: diff_ratio=%.3f%% < min_change=%.3f%% → %s",
        symbol,
        float(row.get("diff_ratio") or 0) * 100,
        MIN_CHANGE_RATIO * 100,
        word,
    )
    count_skip(stats, key)
    return True


def _sized_entry_qty(
    risk: RiskManager,
    row: pd.Series,
    symbol: str,
    current_price: float,
    confidence: float,
    split_ratio: float,
    lot: int,
    stats: OrderExecutionStats,
    words: tuple[str, str],
) -> int:
    """新規買い・新規ショートの株数。0 株なら数えて 0、確信度が低ければ分けた株数（R-308）。

    words は (0 株のときの語, 分けたときの語)。買いは ("スキップ", "発注")。
    """
    kelly_win_rate, kelly_avg_win, kelly_avg_loss = _resolve_kelly_params(
        str(row["market"]), symbol
    )
    qty = risk.calc_position_size(
        symbol,
        current_price,
        win_rate=kelly_win_rate,
        avg_win=kelly_avg_win,
        avg_loss=kelly_avg_loss,
        confidence_ratio=confidence,
    )
    if qty <= 0:
        logger.info(f"[exec] {symbol}: 発注株数 0 → {words[0]}（{risk.zero_reason}）")
        count_skip(stats, zero_qty_skip_key(risk.zero_reason))
        return 0
    if split_ratio < 1.0:
        qty = _apply_split_qty(qty, split_ratio, lot=lot)
        logger.info(
            "[exec] %s: confidence_ratio=%.3f → %.0f%%%s %d株 (R-308)",
            symbol,
            confidence,
            split_ratio * 100,
            words[1],
            qty,
        )
    return qty
