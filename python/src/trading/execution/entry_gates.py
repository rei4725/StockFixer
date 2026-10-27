"""新規（買い・ショート）にだけ使うゲートの評価（#1281）。

DD の基準の更新・リスクのゲート・相関ゲートが落ちた日は、新規を止めたまま決済を続け、
呼び出し元が最後に同じ例外を投げ直す（16:30 の道の #1221 と同じ形）。
"""

from dataclasses import replace

from config.settings import CORRELATION_ENC_THRESHOLD
from src.trading.risk_manager import RiskManager
from src.trading.types import CorrelationGateResult, TradingGateStatus
from src.utils.logger import get_logger

from .stats import OrderExecutionStats

logger = get_logger(__name__)

_GATE_ERROR_REASON = "新規にだけ使うゲートを評価できない"


def _evaluate_entry_gate(
    risk: RiskManager, stats: OrderExecutionStats
) -> tuple[TradingGateStatus, Exception | None]:
    """DD の基準を更新し、当日の新規の可否を返す。落ちたら不合格と例外を返す。"""
    try:
        risk.update_peak_balance()  # R-307: DD基準値を発注前に更新
        gate_status = risk.evaluate_trading_gate()
    except Exception as exc:
        logger.error(
            "[exec] DD の基準の更新かリスクのゲートに失敗。新規は止め、決済は続ける",
            exc_info=True,
        )
        gate = TradingGateStatus(is_allowed=False, stop_active=False, reason=_GATE_ERROR_REASON)
        return gate, exc
    # --- 当日取引可否: 不合格なら新規（買い・ショート）だけ止め、決済は続ける（#874） ---
    if not gate_status.is_allowed:
        logger.warning(
            "[exec] リスクチェック不合格 → 本日の新規発注をスキップ（決済は続ける）: %s",
            gate_status.reason or "理由未設定",
        )
        stats.update(
            {
                "trading_stopped": gate_status.stop_active,
                "stop_reason": gate_status.reason,
                "reason_code": gate_status.reason_code,
            }
        )
    return gate_status, None


def _block_entries(
    gate_status: TradingGateStatus,
) -> tuple[TradingGateStatus, CorrelationGateResult]:
    """ゲートの評価が落ちた日の、新規（買い・ショート）を止めるゲートの組を返す。"""
    reason = gate_status.reason or _GATE_ERROR_REASON
    corr_gate = CorrelationGateResult(
        is_allowed=False,
        enc=0.0,
        enc_threshold=CORRELATION_ENC_THRESHOLD,
        avg_correlation=0.0,
        n_symbols=0,
        reason=_GATE_ERROR_REASON,
    )
    return replace(gate_status, is_allowed=False, reason=reason), corr_gate
