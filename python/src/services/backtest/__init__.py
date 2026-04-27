"""
バックテストドメイン（Backtest Domain）

単一・Walk-Forward・ポートフォリオバックテスト、戦略最適化に関する公開 API。
実装本体は src.backtest BC に移動済み。

# 利用例
    from src.services.backtest import run_backtest_single, run_portfolio_backtest
"""

# 後方互換 re-export — フェーズ4で削除予定（実装本体は src.backtest.* に移動済み）
from src.backtest.optimizer import run_optimize_batch  # noqa: F401
from src.backtest.pipeline import (  # noqa: F401
    load_features,
    print_backtest_metrics,
    run_backtest_single,
    run_backtest_walk_forward,
    save_backtest_results,
)
from src.backtest.portfolio import (  # noqa: F401
    plot_portfolio,
    print_portfolio_metrics,
    run_portfolio_backtest,
    save_portfolio_results,
)
from src.backtest.walk_forward_report import run_walk_forward_comparison_report  # noqa: F401

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
