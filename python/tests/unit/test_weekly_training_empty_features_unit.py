"""特徴量の読み込みが失敗・空の週に、週次学習を成功と見せないことのテスト（#1243）。"""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_CHALLENGERS = ["UnifiedStockXGBoost_challenger", "UnifiedStockLightGBM_challenger"]
_PRODUCTIONS = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]


def test_train_unified_model_raises_when_features_empty():
    """stock_features が空なら、学習の失敗として例外を投げる。"""
    from src.prediction.unified_model_pipeline import train_unified_model

    with patch(
        "src.prediction.unified_model_pipeline.load_all_stock_features",
        return_value=pd.DataFrame(),
    ):
        with pytest.raises(RuntimeError, match="stock_features"):
            train_unified_model(model_type="XGBoostModel", model_name="UnifiedStockXGBoost")


def test_train_unified_model_raises_when_ohlcv_empty():
    """多ホライズンの道でも、特徴量が作れなければ例外を投げる。"""
    from src.prediction.unified_model_pipeline import train_unified_model

    with patch(
        "src.prediction.unified_model_pipeline._load_all_ohlcv_for_unified",
        return_value=(pd.DataFrame(), pd.Series(dtype=float)),
    ):
        with pytest.raises(RuntimeError, match="OHLCV"):
            train_unified_model(
                model_type="XGBoostModel", model_name="UnifiedStockXGB_3d", horizon=3
            )


def test_weekly_training_does_not_report_success_when_features_empty(caplog):
    """load_all_stock_features が空を返した週は、完了の知らせを送らず失敗にする。"""
    from src.orchestration.jobs.weekly import run_weekly_training

    mock_completion = MagicMock()
    with (
        patch("src.prediction.shadow_evaluation.evaluate_shadow_models", side_effect=ValueError),
        patch("src.prediction.shadow_evaluation._UNIFIED_PRODUCTION_NAMES", _PRODUCTIONS),
        patch("src.prediction.shadow_evaluation._UNIFIED_CHALLENGER_NAMES", _CHALLENGERS),
        patch(
            "src.prediction.unified_model_pipeline.load_all_stock_features",
            return_value=pd.DataFrame(),
        ),
        patch("src.prediction.prediction_pipeline.run_accuracy_check"),
        patch("src.reporting.discord.discord_utils.send_drift_alert"),
        patch(
            "src.reporting.discord.discord_utils.send_weekly_training_completion",
            mock_completion,
        ),
        patch("src.reporting.discord.discord_utils.send_promotion_result"),
        caplog.at_level(logging.INFO),
    ):
        with pytest.raises(RuntimeError):
            run_weekly_training()

    mock_completion.assert_not_called()
    messages = [r.getMessage() for r in caplog.records]
    assert not any("Challenger 学習完了" in m for m in messages)
    assert not any("週次モデル学習完了" in m for m in messages)
