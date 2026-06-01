"""reporting BC が外部 BC に依存せず使用するポート型定義。"""

from __future__ import annotations

from typing import Any, Callable, Optional

PredictSingleFn = Callable[[str, str], Optional[Any]]
ExplainShapFn = Callable[[str, str, int], Optional[dict]]
