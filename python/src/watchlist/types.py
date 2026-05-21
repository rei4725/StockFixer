"""
watchlist BC の型定義。

ウォッチリスト管理・バッチ実行で使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SymbolTask:
    """バッチ処理の単位タスク（market + symbol + horizon）。

    batch_runner.load_target_symbols() / run_parallel() で使用する。
    """

    market: str
    symbol: str
    horizon: int = 1


@dataclass
class BatchFailure:
    """バッチ実行の単一失敗エントリ（例外 / タイムアウト起因）。"""

    market: str
    symbol: str
    error: str


@dataclass
class BatchResult:
    """run_parallel() の集約結果。

    succeeded: 正常完了（status=="success"）した結果オブジェクトのリスト
    failed:    例外・タイムアウト・status=="error" による失敗エントリのリスト
    skipped:   status=="skip" として返された結果オブジェクトのリスト
    """

    succeeded: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
