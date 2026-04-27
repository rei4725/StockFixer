"""
売買・リスクドメイン（Trading Domain）

注文実行・リスク管理に関する公開 API。
DDD への移行を見据えた境界コンテキストの入口。

# 利用例
    from src.services.trading import run_daily_orders, RiskManager
"""

from src.services.trading.order_execution_pipeline import run_daily_orders  # noqa: F401
from src.services.trading.risk_manager import RiskError, RiskManager  # noqa: F401

__all__ = [
    "run_daily_orders",
    "RiskManager",
    "RiskError",
]
