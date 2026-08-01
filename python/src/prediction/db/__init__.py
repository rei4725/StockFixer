"""prediction.db パッケージ — 予測結果・モデル精度・精度追跡データの CRUD 操作。

Issue #497: 肥大化した db.py をテーブル/関心別モジュールに分割。
public API は本 __init__ で再公開し、`from src.prediction.db import X` および
src.utils.db の遅延ロード（importlib.import_module 経由 getattr）の後方互換を維持する。
"""

from src.utils.db._connection import _db_connection  # noqa: F401  patch target 後方互換

from .accuracy import (  # noqa: F401
    load_drift_summary,
    load_prediction_accuracy,
    load_top_prediction_misses,
    load_weekly_accuracy_snapshots,
    save_prediction_accuracy,
    save_weekly_accuracy_snapshot,
)
from .features import (  # noqa: F401
    load_excluded_features,
    load_feature_exclusion_candidates,
    load_shap_latest,
    save_feature_selection,
    save_shap_values,
)
from .model_metrics import load_model_weights, save_model_metrics  # noqa: F401
from .order_summary import load_turnover_comparison, save_order_run_summary  # noqa: F401
from .paper_real_diff import (  # noqa: F401
    load_open_close_advantage_summary,
    load_paper_real_diff_summary,
    upsert_paper_real_diff,
)
from .prediction_results import (  # noqa: F401
    load_latest_prediction_timestamp,
    load_prediction_markets,
    load_prediction_results,
    load_previous_run_stats,
    load_shadow_comparison,
    save_prediction_results,
)

__all__ = [
    # prediction_results
    "save_prediction_results",
    "load_latest_prediction_timestamp",
    "load_prediction_markets",
    "load_prediction_results",
    "load_previous_run_stats",
    "load_shadow_comparison",
    # paper_real_diff
    "upsert_paper_real_diff",
    "load_paper_real_diff_summary",
    "load_open_close_advantage_summary",
    # model_metrics
    "save_model_metrics",
    "load_model_weights",
    # accuracy
    "save_prediction_accuracy",
    "load_prediction_accuracy",
    "load_top_prediction_misses",
    "load_drift_summary",
    "save_weekly_accuracy_snapshot",
    "load_weekly_accuracy_snapshots",
    # features
    "save_shap_values",
    "load_shap_latest",
    "save_feature_selection",
    "load_feature_exclusion_candidates",
    "load_excluded_features",
    # order_run_summary
    "save_order_run_summary",
    "load_turnover_comparison",
]
