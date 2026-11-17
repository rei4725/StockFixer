"""
統合モデルの特徴量選択

permutation importance を測り、統合モデルの特徴量選択の結果を書く（#1968 で
unified_model_pipeline.py から中身を変えずに移した）。
"""

from datetime import datetime

import pandas as pd
from sklearn.inspection import permutation_importance

from config.settings import (
    FEATURE_SELECTION_DROP_RATIO,
    FEATURE_SELECTION_MIN_FEATURES,
    PERMUTATION_IMPORTANCE_REPEATS,
)
from src.prediction.db import save_feature_selection


def _compute_and_save_unified_feature_selection(model, X: pd.DataFrame, y: pd.Series) -> None:
    if X.empty or len(X.columns) <= FEATURE_SELECTION_MIN_FEATURES:
        return

    sample_size = min(len(X), 200)
    X_eval = X.tail(sample_size)
    y_eval = y.reindex(X_eval.index)
    result = permutation_importance(
        model.model,
        X_eval,
        y_eval,
        n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
        random_state=42,
        scoring="neg_mean_squared_error",
    )
    selection_df = (
        pd.DataFrame(
            {
                "feature": X_eval.columns.tolist(),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values(["importance_mean", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selection_df["importance_rank"] = range(1, len(selection_df) + 1)
    max_exclusions = max(0, len(selection_df) - FEATURE_SELECTION_MIN_FEATURES)
    exclude_count = min(int(len(selection_df) * FEATURE_SELECTION_DROP_RATIO), max_exclusions)
    excluded = (
        set(selection_df.tail(exclude_count)["feature"].tolist()) if exclude_count > 0 else set()
    )
    selection_df["is_excluded"] = selection_df["feature"].isin(excluded)
    selection_df["protected_by_shap"] = False
    save_feature_selection(
        "unified",
        "all",
        model.model_name,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        selection_df,
    )
