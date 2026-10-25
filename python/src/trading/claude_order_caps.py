"""
ClaudeTrader の1回（run）の中で出した新規の買いを数え、上限で止める。

上限をシステムプロンプトに書くだけでは Claude が守る保証がないため、
place_order のハンドラーから Python 側で強制する。
数え方は 8:50 の道（execution/runner.py の position_count と head(MAX_ORDERS_PER_RUN)）に合わせる。
ペーパーの買いは約定まで保有に出ないため、broker.get_positions() だけでは同じ回の買いを数えられない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import MAX_ORDERS_PER_RUN, MAX_POSITIONS


@dataclass
class RunBuyCaps:
    """1回の中の新規の買いの数・建玉の数・買った銘柄を持つ"""

    position_count: int
    buy_orders: int = 0
    bought_symbols: set[str] = field(default_factory=set)

    def check_buy(self, symbol: str) -> str | None:
        """買いを止める理由を返す。止めないなら None"""
        if symbol in self.bought_symbols:
            return f"{symbol} はこの回で既に買い発注済み"
        if self.buy_orders >= MAX_ORDERS_PER_RUN:
            return f"1回の新規買いが MAX_ORDERS_PER_RUN={MAX_ORDERS_PER_RUN} 件に到達"
        if self.position_count >= MAX_POSITIONS:
            return f"保有上限 MAX_POSITIONS={MAX_POSITIONS} に到達"
        return None

    def record_buy(self, symbol: str) -> None:
        """発注できた買いを数える"""
        self.buy_orders += 1
        self.position_count += 1
        self.bought_symbols.add(symbol)
