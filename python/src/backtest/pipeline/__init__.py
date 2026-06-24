"""backtest.pipeline パッケージ — バックテスト実行ロジックの統合サービス層。

Issue #511: 肥大化した pipeline.py を責務別モジュールに分割。
public API は本 __init__ で再公開し、`from src.backtest.pipeline import X` の
後方互換を維持する。

- features.py   … 特徴量ロード (load_features)
- runner.py     … 実行ロジック (run_backtest_single / run_backtest_walk_forward)
- reporting.py  … 結果保存・表示・ベンチマーク比較
"""

from src.backtest.ports import get_model_manager  # noqa: F401  patch target 後方互換

from .features import load_features  # noqa: F401
from .reporting import (  # noqa: F401
    fetch_benchmark_for_result,
    plot_backtest_chart,
    print_backtest_metrics,
    save_backtest_results,
)
from .runner import (  # noqa: F401
    _build_task,
    _ensemble_predict,
    run_backtest_single,
    run_backtest_walk_forward,
)

__all__ = [
    "load_features",
    "run_backtest_single",
    "run_backtest_walk_forward",
    "save_backtest_results",
    "print_backtest_metrics",
    "plot_backtest_chart",
    "fetch_benchmark_for_result",
]
