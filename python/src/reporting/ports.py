"""reporting BC が外部 BC に依存せず使用するポート型定義。"""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.domain.ports import AnalyticsQuery

PredictSingleFn = Callable[[str, str], Optional[Any]]
ExplainShapFn = Callable[[str, str, int], Optional[dict]]


# ============================================
# AnalyticsQuery の注入口（Discord Bot 用）
# ============================================
#
# Discord Bot はモジュール直下の `bot` をコマンドハンドラごと起動するため、
# ハンドラ呼び出しの途中でポートを引数として渡す経路が無い。
# 合成ルート（orchestration.port_wiring.wire_ports()）がここへ注入する。

_analytics_query: Optional[AnalyticsQuery] = None


def set_analytics_query(query: AnalyticsQuery) -> None:
    """テストや orchestration から実装を注入するためのセッター。"""
    global _analytics_query
    _analytics_query = query


def get_analytics_query() -> AnalyticsQuery:
    """注入済みの AnalyticsQuery を返す。未注入なら RuntimeError。

    注入は orchestration の合成ルートで行う:
    `src.orchestration.port_wiring.wire_ports()`（エントリポイント起動時に呼ぶ）。
    """
    if _analytics_query is None:
        raise RuntimeError(
            "AnalyticsQuery が未注入です。エントリポイントで "
            "src.orchestration.port_wiring.wire_ports() を呼ぶか、"
            "set_analytics_query() で実装を注入してください。"
        )
    return _analytics_query
