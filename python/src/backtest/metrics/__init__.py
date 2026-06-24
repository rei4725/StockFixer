"""backtest.metrics パッケージ — バックテスト評価指標。

Issue #511: 肥大化した metrics.py を責務別モジュールに分割。
public/internal API は本 __init__ で再公開し、`from src.backtest.metrics import X`
の後方互換を維持する。

- core.py:        compute_metrics / コスト比較 / レジーム別 / Sharpe・DD ヘルパ
- overfitting.py: DSR / PBO (CSCV) / モンテカルロ equity
- reporting.py:   plot_backtest / fetch_benchmark_returns / BENCHMARK_TICKERS
"""

from src.backtest.metrics.core import (  # noqa: F401
    _annualize_sharpe,
    _empty_metrics,
    _estimate_trades_per_year,
    _extract_trade_pnl,
    _max_drawdown,
    _sharpe_per_trade,
    compute_cost_comparison_metrics,
    compute_metrics,
    compute_metrics_by_regime,
)
from src.backtest.metrics.overfitting import (  # noqa: F401
    _sharpe_by_column,
    deflated_sharpe_ratio,
    monte_carlo_equity,
    probability_of_backtest_overfitting,
)
from src.backtest.metrics.reporting import (  # noqa: F401
    BENCHMARK_TICKERS,
    fetch_benchmark_returns,
    plot_backtest,
)

__all__ = [
    "compute_metrics",
    "compute_cost_comparison_metrics",
    "compute_metrics_by_regime",
    "_empty_metrics",
    "_extract_trade_pnl",
    "_sharpe_per_trade",
    "_estimate_trades_per_year",
    "_annualize_sharpe",
    "_max_drawdown",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "_sharpe_by_column",
    "monte_carlo_equity",
    "plot_backtest",
    "fetch_benchmark_returns",
    "BENCHMARK_TICKERS",
]
