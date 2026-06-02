"""backtest BC が外部 BC に依存せず使用するポート定義。"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import numpy as np
import pandas as pd


class BacktestModelManagerPort(Protocol):
    """バックテストが必要とする ModelManager の最小インターフェース。"""

    def create_model(self, model_type: str, model_name: str) -> None: ...
    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict_with_model(self, model_name: str, X: pd.DataFrame) -> np.ndarray: ...


_model_manager_factory: Optional[Callable[[], Any]] = None


def set_model_manager_factory(factory: Callable[[], Any]) -> None:
    """ModelManager ファクトリを登録する（entry point / orchestration 起動時に1回だけ呼ぶ）。"""
    global _model_manager_factory
    _model_manager_factory = factory


def get_model_manager() -> BacktestModelManagerPort:
    """登録済みファクトリで ModelManager インスタンスを生成して返す。"""
    if _model_manager_factory is None:
        raise RuntimeError(
            "model_manager_factory が未設定です。set_model_manager_factory() を先に呼び出してください。"
        )
    return _model_manager_factory()  # type: ignore[return-value]
