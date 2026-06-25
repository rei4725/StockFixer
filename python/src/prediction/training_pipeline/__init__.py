"""銘柄別モデル学習パイプラインサービス

指定した銘柄のデータを使い、XGBoost・LightGBMモデルを学習・保存する。

Issue #511: 肥大化した training_pipeline.py を責務別モジュールへ分割。
public API は本 __init__ で再公開し、`from src.prediction.training_pipeline import X`
の import 後方互換を維持する。

- _features.py:  特徴量の読み込み・前処理・SHAP/permutation 寄与分析
- _training.py:  学習オーケストレーション（単一銘柄・バッチ・複数ホライズン）
"""

from ._features import (  # noqa: F401
    _apply_feature_exclusions,
    _build_feature_selection_frame,
    _compute_and_save_permutation_importance,
    _compute_and_save_shap,
    _extract_mean_abs_shap_values,
    _mask_earnings_rows,
    load_features_for_training,
)
from ._training import (  # noqa: F401
    _compute_purged_cv_metrics,
    _compute_training_metrics,
    _log_training_summary,
    _train_models_for_horizon,
    run_batch_training,
    train_all_models,
    train_models_for_symbol,
    train_models_for_symbol_task,
)

__all__ = [
    # _features
    "load_features_for_training",
    "_apply_feature_exclusions",
    "_mask_earnings_rows",
    "_extract_mean_abs_shap_values",
    "_build_feature_selection_frame",
    "_compute_and_save_permutation_importance",
    "_compute_and_save_shap",
    # _training
    "train_models_for_symbol",
    "train_models_for_symbol_task",
    "run_batch_training",
    "train_all_models",
    "_train_models_for_horizon",
    "_compute_training_metrics",
    "_compute_purged_cv_metrics",
    "_log_training_summary",
]
