"""週次学習の [1/4] 影の比べと [2/4] 昇格ゲートの失敗を知らせることのテスト（#1295）。"""

import logging
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

_CHALLENGERS = ["UnifiedStockXGBoost_challenger", "UnifiedStockLightGBM_challenger"]
_PRODUCTIONS = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
_WINS = {
    "challenger_wins": True,
    "production_hit_rate": 0.5,
    "production_sharpe": 0.5,
    "challenger_hit_rate": 0.6,
    "challenger_sharpe": 0.8,
}


def _enter_patches(stack: ExitStack, tmp_path, shadow, promotion, mock_train, mock_send):
    (tmp_path / "unified").mkdir(exist_ok=True)
    stack.enter_context(patch("config.settings.AUTO_PROMOTE_MODEL", False))
    stack.enter_context(patch("src.prediction.promotion_gate.evaluate_promotion", promotion))
    stack.enter_context(patch("src.prediction.promotion_gate.save_promotion_result"))
    stack.enter_context(patch("src.prediction.shadow_evaluation.evaluate_shadow_models", shadow))
    stack.enter_context(patch("src.prediction.shadow_evaluation.promote_unified_challenger"))
    stack.enter_context(
        patch("src.prediction.shadow_evaluation._UNIFIED_PRODUCTION_NAMES", _PRODUCTIONS)
    )
    stack.enter_context(
        patch("src.prediction.shadow_evaluation._UNIFIED_CHALLENGER_NAMES", _CHALLENGERS)
    )
    stack.enter_context(
        patch("src.prediction.unified_model_pipeline.train_unified_model", mock_train)
    )
    stack.enter_context(
        patch("src.utils.data_path_utils.get_models_dir", return_value=str(tmp_path))
    )
    stack.enter_context(patch("src.prediction.prediction_pipeline.run_accuracy_check"))
    stack.enter_context(patch("src.reporting.discord.discord_utils.send_drift_alert"))
    stack.enter_context(
        patch("src.reporting.discord.discord_utils.send_weekly_training_completion")
    )
    stack.enter_context(patch("src.reporting.discord.discord_utils.send_promotion_result"))
    stack.enter_context(patch("src.reporting.discord.discord_utils.send_webhook_text", mock_send))


def test_shadow_evaluation_error_notifies_once_and_retrains(tmp_path, caplog):
    """[1/4] が例外なら1回知らせ、初回実行と出さず、[4/4] は続ける。"""
    from src.orchestration.jobs.weekly import run_weekly_training

    shadow = MagicMock(side_effect=RuntimeError("prediction_accuracy が読めない"))
    mock_train = MagicMock()
    mock_send = MagicMock()
    with ExitStack() as stack:
        _enter_patches(stack, tmp_path, shadow, MagicMock(), mock_train, mock_send)
        with caplog.at_level(logging.INFO):
            run_weekly_training()

    assert mock_send.call_count == 1
    text = mock_send.call_args[0][0]
    assert "[1/4]" in text and "prediction_accuracy が読めない" in text
    assert not any("初回実行" in r.getMessage() for r in caplog.records)
    assert mock_train.call_count == 2
    assert shadow.call_args.kwargs["raise_on_error"] is True


def test_promotion_gate_error_notifies_once_and_retrains(tmp_path):
    """challenger_wins=True で [2/4] が例外なら1回知らせ、[4/4] は続ける。"""
    from src.orchestration.jobs.weekly import run_weekly_training

    promotion = MagicMock(side_effect=RuntimeError("昇格ゲートの集計失敗"))
    mock_train = MagicMock()
    mock_send = MagicMock()
    with ExitStack() as stack:
        _enter_patches(
            stack, tmp_path, MagicMock(return_value=_WINS), promotion, mock_train, mock_send
        )
        run_weekly_training()

    promotion.assert_called_once()
    assert mock_send.call_count == 1
    text = mock_send.call_args[0][0]
    assert "[2/4]" in text and "昇格ゲートの集計失敗" in text
    assert mock_train.call_count == 2


def test_no_failure_notice_when_both_stages_pass(tmp_path):
    """[1/4] と [2/4] が通った週は、失敗の知らせを出さない。"""
    from src.orchestration.jobs.weekly import run_weekly_training
    from src.prediction.promotion_gate import PromotionResult

    gate = PromotionResult(
        shadow_model="challenger",
        current_model="production",
        evaluated_at="2026-10-27T00:00:00",
        eligible=False,
        require_manual_approval=True,
        criteria={},
        reason="未達",
    )
    mock_send = MagicMock()
    with ExitStack() as stack:
        _enter_patches(
            stack,
            tmp_path,
            MagicMock(return_value=_WINS),
            MagicMock(return_value=gate),
            MagicMock(),
            mock_send,
        )
        run_weekly_training()

    mock_send.assert_not_called()


def _failing_db(stack: ExitStack) -> None:
    stack.enter_context(patch("src.prediction.db.accuracy._db_connection"))
    stack.enter_context(
        patch("src.prediction.db.accuracy.pd.read_sql", side_effect=RuntimeError("DB down"))
    )
    stack.enter_context(patch("src.prediction.shadow_evaluation._record_ab_result"))


def test_evaluate_shadow_models_raises_on_read_error_when_asked():
    """週次学習の道（raise_on_error=True）は、読み込みの例外を投げる。"""
    from src.prediction.shadow_evaluation import evaluate_shadow_models

    with ExitStack() as stack:
        _failing_db(stack)
        with pytest.raises(RuntimeError, match="DB down"):
            evaluate_shadow_models(raise_on_error=True)


def test_evaluate_shadow_models_default_keeps_empty_on_read_error():
    """raise_on_error を渡さない呼び出し元は、今までどおり空として扱う。"""
    from src.prediction.shadow_evaluation import evaluate_shadow_models

    with ExitStack() as stack:
        _failing_db(stack)
        result = evaluate_shadow_models()

    assert result["challenger_wins"] is False
    assert result["n_production"] == 0
    assert result["n_challenger"] == 0
