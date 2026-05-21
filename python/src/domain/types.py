"""
ドメイン共有型定義。

複数の Bounded Context から参照される汎用型を集約する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SymbolTask:
    """バッチ処理の単位タスク（market + symbol + horizon）。

    batch_runner.load_target_symbols() / run_parallel() で使用する。
    複数 BC から参照される共有型のため domain/ に配置する。
    """

    market: str
    symbol: str
    horizon: int = 1
