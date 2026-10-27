"""
ClaudeTrader の1回（run）の中で出した新規の買いを数え、上限で止める。

上限をシステムプロンプトに書くだけでは Claude が守る保証がないため、
place_order のハンドラーから Python 側で強制する。
数え方は 8:50 の道（execution/runner.py の position_count と head(MAX_ORDERS_PER_RUN)）に合わせる。
ペーパーの買いは約定まで保有に出ないため、broker.get_positions() だけでは同じ回の買いを数えられない。
相関ゲートと同一業種の上限も 8:50 の道（runner.py の evaluate_correlation_gate と
execution/selection.py の _apply_buy_sector_limit）と同じく、回の始めの保有から当てる（#1153）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from config.settings import (
    CORRELATION_ENC_THRESHOLD,
    CORRELATION_WINDOW_DAYS,
    MAX_ORDERS_PER_RUN,
    MAX_POSITIONS,
    MAX_SECTOR_POSITIONS,
)
from src.trading.correlation_risk import evaluate_correlation_gate
from src.utils.logger import get_logger
from src.utils.sector_constraints import get_symbol_sector

logger = get_logger(__name__)


@dataclass
class RunBuyCaps:
    """1回の中の新規の買いの数・建玉の数・買った銘柄を持つ

    sector_counts が None のときは業種の上限を当てない。
    corr_block_reason が入っているときは相関ゲート不合格として新規の買いを止める。
    """

    position_count: int
    buy_orders: int = 0
    bought_symbols: set[str] = field(default_factory=set)
    market: str = "jp"
    corr_block_reason: str | None = None
    sector_counts: Counter[str] | None = None

    def _sector(self, symbol: str) -> str:
        return get_symbol_sector(self.market, symbol.replace(".T", ""))

    def check_buy(self, symbol: str) -> str | None:
        """買いを止める理由を返す。止めないなら None"""
        if symbol in self.bought_symbols:
            return f"{symbol} はこの回で既に買い発注済み"
        if self.corr_block_reason is not None:
            return f"相関ゲート不合格のため新規買いを停止: {self.corr_block_reason}"
        if self.buy_orders >= MAX_ORDERS_PER_RUN:
            return f"1回の新規買いが MAX_ORDERS_PER_RUN={MAX_ORDERS_PER_RUN} 件に到達"
        if self.position_count >= MAX_POSITIONS:
            return f"保有上限 MAX_POSITIONS={MAX_POSITIONS} に到達"
        if self.sector_counts is not None and MAX_SECTOR_POSITIONS > 0:
            sector = self._sector(symbol)
            if self.sector_counts[sector] >= MAX_SECTOR_POSITIONS:
                return (
                    f"{symbol} の業種 {sector} が保有と同じ回の買いで "
                    f"MAX_SECTOR_POSITIONS={MAX_SECTOR_POSITIONS} に到達"
                )
        return None

    def record_buy(self, symbol: str) -> None:
        """発注できた買いを数える"""
        self.buy_orders += 1
        self.position_count += 1
        self.bought_symbols.add(symbol)
        if self.sector_counts is not None:
            self.sector_counts[self._sector(symbol)] += 1


def start_run_buy_caps(
    held_symbols: Iterable[str], market: str, short_count: int = 0
) -> RunBuyCaps:
    """回の始めの保有から、相関ゲートを1回だけ評価し、業種の数を数えた RunBuyCaps を作る

    short_count は空売りの建玉の数で、建玉の数（MAX_POSITIONS）にだけ足す（#1211）。
    業種の上限と相関ゲートには入れない。
    """
    held_codes = sorted({s.replace(".T", "") for s in held_symbols})
    corr_gate = evaluate_correlation_gate(
        held_codes,
        market,
        window=CORRELATION_WINDOW_DAYS,
        enc_threshold=CORRELATION_ENC_THRESHOLD,
    )
    corr_block_reason = None
    if not corr_gate.is_allowed:
        corr_block_reason = corr_gate.reason or "ENC閾値未満"
        logger.warning("[claude_agent] 相関ゲート不合格のため新規買いを停止: %s", corr_block_reason)
    return RunBuyCaps(
        position_count=len(held_codes) + short_count,
        market=market,
        corr_block_reason=corr_block_reason,
        sector_counts=Counter(get_symbol_sector(market, s) for s in held_codes),
    )
