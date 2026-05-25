"""
watchlist BC の型定義。

ウォッチリスト管理・バッチ実行で使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.types import BatchFailure as BatchFailure  # noqa: F401
from src.domain.types import BatchResult as BatchResult  # noqa: F401


@dataclass
class SymbolTask:
    """バッチ処理の単位タスク（market + symbol + horizon）。

    batch_runner.load_target_symbols() / run_parallel() で使用する。
    """

    market: str
    symbol: str
    horizon: int = 1
