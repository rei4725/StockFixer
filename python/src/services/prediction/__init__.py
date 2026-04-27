"""
予測ドメイン（Prediction Domain）

株価予測・精度チェック・シャドウ評価に関する公開 API。
DDD への移行を見据えた境界コンテキスト（Bounded Context）の入口。

# 利用例
    from src.services.prediction import predict_all_unified, run_accuracy_check
"""

from src.services.prediction.prediction_pipeline import (  # noqa: F401
    find_model_files,
    get_optimal_params,
    output_top_worst_results,
    predict_all_individual,
    predict_all_unified,
    predict_all_unified_multi_horizon,
    run_accuracy_check,
)
from src.services.prediction.shadow_evaluation_pipeline import (  # noqa: F401
    evaluate_shadow_models,
    run_shadow_prediction,
)
from src.services.prediction.unified_model_pipeline import train_unified_model  # noqa: F401

__all__ = [
    # 予測
    "predict_all_unified",
    "predict_all_individual",
    "predict_all_unified_multi_horizon",
    "output_top_worst_results",
    # 精度
    "run_accuracy_check",
    "find_model_files",
    "get_optimal_params",
    # 統合モデル学習
    "train_unified_model",
    # シャドウ評価
    "evaluate_shadow_models",
    "run_shadow_prediction",
]
