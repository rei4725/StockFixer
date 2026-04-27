"""
バックテストドメイン（Backtest Domain）

単一・Walk-Forward・ポートフォリオバックテスト、戦略最適化に関する公開 API。
DDD への移行を見据えた境界コンテキストの入口。

# 利用例
    from src.services.backtest import run_backtest_single, run_portfolio_backtest
"""

from src.services.backtest.backtest_optimize_pipeline import run_optimize_batch  # noqa: F401
from src.services.backtest.backtest_pipeline import (  # noqa: F401
    load_features,
    print_backtest_metrics,
    run_backtest_single,
    run_backtest_walk_forward,
    save_backtest_results,
)
from src.services.backtest.portfolio_backtest import (  # noqa: F401
    plot_portfolio,
    print_portfolio_metrics,
    run_portfolio_backtest,
    save_portfolio_results,
)
from src.services.backtest.walk_forward_report_pipeline import (  # noqa: F401
    run_walk_forward_comparison_report,
)

__all__ = [
    # 単一バックテスト
    "run_backtest_single",
    "run_backtest_walk_forward",
    "load_features",
    "save_backtest_results",
    "print_backtest_metrics",
    # 最適化
    "run_optimize_batch",
    # ポートフォリオ
    "run_portfolio_backtest",
    "save_portfolio_results",
    "plot_portfolio",
    "print_portfolio_metrics",
    # Walk-Forward レポート
    "run_walk_forward_comparison_report",
]
