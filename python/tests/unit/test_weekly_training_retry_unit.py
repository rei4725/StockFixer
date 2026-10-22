"""週の学習を同じ週にやり直したときに、昇格を重ねないことのテスト（#905）。"""

import logging
import os
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.prediction.promotion_gate import PromotionResult

_CHALLENGERS = ["UnifiedStockXGBoost_challenger", "UnifiedStockLightGBM_challenger"]
_PRODUCTIONS = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]


def _eligible_result() -> PromotionResult:
    return PromotionResult(
        shadow_model="challenger",
        current_model="production",
        evaluated_at="2026-10-20T00:00:00",
        eligible=True,
        require_manual_approval=False,
        criteria={},
        reason="全基準クリア。自動昇格可能。",
    )


def _write_challenger(unified_dir: str, name: str, mtime: float) -> None:
    path = os.path.join(unified_dir, f"{name}.joblib")
    with open(path, "wb") as f:
        f.write(b"model")
    os.utime(path, (mtime, mtime))


@pytest.fixture
def unified_dir(tmp_path):
    d = tmp_path / "unified"
    d.mkdir()
    # 先週の学習で作られ、1週の影の予想を経た challenger
    last_week = (datetime.now(timezone.utc) - timedelta(days=8)).timestamp()
    for name in _CHALLENGERS:
        _write_challenger(str(d), name, last_week)
    return d


def _enter_common_patches(stack: ExitStack, models_dir: str, mock_promote, mock_train):
    stack.enter_context(patch("config.settings.AUTO_PROMOTE_MODEL", True))
    stack.enter_context(
        patch("src.prediction.promotion_gate.evaluate_promotion", return_value=_eligible_result())
    )
    stack.enter_context(patch("src.prediction.promotion_gate.save_promotion_result"))
    stack.enter_context(
        patch(
            "src.prediction.shadow_evaluation.evaluate_shadow_models",
            return_value={
                "challenger_wins": True,
                "production_hit_rate": 0.5,
                "production_sharpe": 0.5,
                "challenger_hit_rate": 0.6,
                "challenger_sharpe": 0.8,
            },
        )
    )
    stack.enter_context(
        patch("src.prediction.shadow_evaluation.promote_unified_challenger", mock_promote)
    )
    stack.enter_context(
        patch("src.prediction.shadow_evaluation._UNIFIED_PRODUCTION_NAMES", _PRODUCTIONS)
    )
    stack.enter_context(
        patch("src.prediction.shadow_evaluation._UNIFIED_CHALLENGER_NAMES", _CHALLENGERS)
    )
    stack.enter_context(
        patch("src.prediction.unified_model_pipeline.train_unified_model", mock_train)
    )
    stack.enter_context(patch("src.utils.data_path_utils.get_models_dir", return_value=models_dir))
    stack.enter_context(patch("src.prediction.prediction_pipeline.run_accuracy_check"))
    stack.enter_context(patch("src.reporting.discord.discord_utils.send_drift_alert"))
    stack.enter_context(
        patch("src.reporting.discord.discord_utils.send_weekly_training_completion")
    )
    stack.enter_context(patch("src.reporting.discord.discord_utils.send_promotion_result"))


def test_first_run_of_week_promotes(unified_dir):
    """同じ週の最初の回は、今までどおり昇格する。"""
    from src.orchestration.jobs.weekly import run_weekly_training

    mock_promote = MagicMock(return_value={"promoted": _PRODUCTIONS, "skipped": []})
    with ExitStack() as stack:
        _enter_common_patches(stack, str(unified_dir.parent), mock_promote, MagicMock())
        run_weekly_training()

    mock_promote.assert_called_once()


def test_retry_after_challenger_training_failure_does_not_promote_again(unified_dir, caplog):
    """1回目で昇格し、challenger の LightGBM の学習で落ちた後のやり直しでは昇格しない。"""
    from src.orchestration.jobs.weekly import run_weekly_training

    mock_promote = MagicMock(return_value={"promoted": _PRODUCTIONS, "skipped": []})

    def _train(model_type, model_name):
        if model_type == "LightGBMModel":
            raise MemoryError("LightGBM の学習でメモリ不足")
        # XGBoost の challenger は学習して保存し直される
        _write_challenger(str(unified_dir), model_name, time.time())

    mock_train = MagicMock(side_effect=_train)
    with ExitStack() as stack:
        _enter_common_patches(stack, str(unified_dir.parent), mock_promote, mock_train)
        with pytest.raises(MemoryError):
            run_weekly_training()
        assert mock_promote.call_count == 1

        # スケジューラのやり直し（同じ週の2回目）
        with caplog.at_level(logging.WARNING):
            with pytest.raises(MemoryError):
                run_weekly_training()

    assert mock_promote.call_count == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("昇格を飛ばす" in r.getMessage() for r in warnings)
    assert any("UnifiedStockXGBoost_challenger" in r.getMessage() for r in warnings)


def test_missing_challenger_files_do_not_block_promotion(tmp_path):
    """challenger のファイルが無いときは、この確かめでは止めない（従来どおり）。"""
    from src.orchestration.jobs.weekly import run_weekly_training

    (tmp_path / "unified").mkdir()
    mock_promote = MagicMock(return_value={"promoted": [], "skipped": _CHALLENGERS})
    with ExitStack() as stack:
        _enter_common_patches(stack, str(tmp_path), mock_promote, MagicMock())
        run_weekly_training()

    mock_promote.assert_called_once()
