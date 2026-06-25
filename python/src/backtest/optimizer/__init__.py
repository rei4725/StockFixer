"""backtest.optimizer パッケージ — バックテスト最適化パイプラインサービス。

閾値・ストップロス・テイクプロフィットのグリッドサーチ（および Optuna 探索）を
Walk-Forward 検証で実行し、最適パラメータを特定する。

Issue #511: 肥大化した optimizer.py を責務別モジュールに分割。
public API は本 __init__ で再公開し、`from src.backtest.optimizer import X`
の後方互換を維持する。

run_backtest_optimize.py / run_backtest_optimize_batch.py はこのパッケージの
関数を呼び出すラッパーとして機能する。
"""

from src.backtest.optimizer._pbo import (  # noqa: F401
    _DSR_WARN_THRESHOLD,
    _PBO_WARN_THRESHOLD,
    _compute_pbo_from_fold_returns,
)
from src.backtest.optimizer.batch import run_optimize_batch, run_optuna_batch  # noqa: F401
from src.backtest.optimizer.grid import _frange, _parse_grid_values, run_optimization  # noqa: F401
from src.backtest.optimizer.optuna_search import run_optuna_optimization  # noqa: F401
from src.backtest.optimizer.persistence import (  # noqa: F401
    get_optimal_params,
    print_optimization_results,
    save_optimal_params_json,
    save_optimization_results,
)

__all__ = [
    # grid
    "run_optimization",
    "_frange",
    "_parse_grid_values",
    # persistence
    "print_optimization_results",
    "save_optimization_results",
    "save_optimal_params_json",
    "get_optimal_params",
    # optuna
    "run_optuna_optimization",
    # batch
    "run_optimize_batch",
    "run_optuna_batch",
    # pbo guard
    "_compute_pbo_from_fold_returns",
    "_DSR_WARN_THRESHOLD",
    "_PBO_WARN_THRESHOLD",
]
