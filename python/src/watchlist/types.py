"""
watchlist BC の型定義。

ウォッチリスト管理・バッチ実行で使用するドメイン型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SymbolTask:
    """バッチ処理の単位タスク（market + symbol + horizon）。

    batch_runner.load_target_symbols() / run_parallel() で使用する。
    """

    market: str
    symbol: str
    horizon: int = 1


@dataclass
class WatchlistPredictionRow:
    """ウォッチリスト表示用の1行。"""

    symbol: str
    current_price: Optional[float]
    avg_pred_price: Optional[float]
    diff_ratio: Optional[float]

    @classmethod
    def to_dataframe(cls, rows: list["WatchlistPredictionRow"]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": row.symbol,
                    "current_price": row.current_price,
                    "avg_pred_price": row.avg_pred_price,
                    "diff_ratio": row.diff_ratio,
                }
                for row in rows
            ]
        )


@dataclass
class WatchlistPredictionView:
    """ウォッチリスト表示の取得結果。"""

    rows: list[WatchlistPredictionRow] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error_message is None


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
