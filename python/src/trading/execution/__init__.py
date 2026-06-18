"""trading.execution パッケージ — 自動発注パイプライン。

Issue #497: 肥大化した execution.py を責務別モジュールに分割。
public/internal API は本 __init__ で再公開し、`from src.trading.execution import X`
の後方互換を維持する。
"""

from config.settings import (  # noqa: F401  re-export for tests/backward-compat
    BUY_THRESHOLD,
    MAX_ORDERS_PER_RUN,
    MIN_CHANGE_RATIO,
)
from src.trading.risk_manager import RiskManager  # noqa: F401  patch target 後方互換

from .params import (  # noqa: F401
    _apply_split_qty,
    _attach_dynamic_thresholds,
    _calc_split_ratio,
    _choose_order_params,
    _compute_market_threshold_scale,
    _load_execution_metrics,
    _resolve_base_threshold,
    _resolve_kelly_params,
)
from .predictions import _load_latest_predictions  # noqa: F401
from .recording import (  # noqa: F401
    _link_paper_order_metadata,
    _record_order,
    _sync_live_execution_diffs,
)
from .runner import run_daily_orders  # noqa: F401
from .selection import (  # noqa: F401
    _apply_buy_sector_limit,
    _compute_ml_exit_signals,
    _determine_entry_horizon,
    _get_held_symbols,
    _load_exit_model,
)
from .sl_tp import _check_sl_tp_exits  # noqa: F401
from .stats import OrderExecutionStats  # noqa: F401

__all__ = [
    "OrderExecutionStats",
    "run_daily_orders",
    "_load_latest_predictions",
    "_compute_market_threshold_scale",
    "_resolve_base_threshold",
    "_attach_dynamic_thresholds",
    "_resolve_kelly_params",
    "_calc_split_ratio",
    "_apply_split_qty",
    "_choose_order_params",
    "_load_execution_metrics",
    "_apply_buy_sector_limit",
    "_get_held_symbols",
    "_load_exit_model",
    "_compute_ml_exit_signals",
    "_determine_entry_horizon",
    "_link_paper_order_metadata",
    "_record_order",
    "_sync_live_execution_diffs",
    "_check_sl_tp_exits",
]
