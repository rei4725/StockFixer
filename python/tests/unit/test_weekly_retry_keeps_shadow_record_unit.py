"""同じ週のやり直しで、前の回の影の比べと昇格の記録を空の結果で上書きしないことのテスト（#1798）。"""

import json
import os
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.prediction.promotion_gate import PromotionResult

_CHALLENGERS = ["UnifiedStockXGBoost_challenger", "UnifiedStockLightGBM_challenger"]
_PRODUCTIONS = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]


def _compared(wins: bool) -> dict:
    """1回目の影の比べ（先週の challenger の窓に行がある）。"""
    return {
        "production_hit_rate": 0.5,
        "production_sharpe": 0.4,
        "challenger_hit_rate": 0.6 if wins else 0.45,
        "challenger_sharpe": 0.8 if wins else 0.1,
        "challenger_wins": wins,
        "n_production": 3000,
        "n_challenger": 2400,
        "challenger_trained_at": "2026-11-02T16:30:00+00:00",
        "production_changed_at": "2026-10-20T16:30:00+00:00",
        "window_days": 5,
    }


def _empty_window() -> dict:
    """この週の学習か昇格で窓が今になり、比べる行が0件になったやり直しの結果。"""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "production_hit_rate": None,
        "production_sharpe": None,
        "challenger_hit_rate": None,
        "challenger_sharpe": None,
        "challenger_wins": False,
        "n_production": 0,
        "n_challenger": 0,
        "challenger_trained_at": now,
        "production_changed_at": now,
        "window_days": 0,
    }


def _eligible_result() -> PromotionResult:
    return PromotionResult(
        shadow_model="challenger",
        current_model="production",
        evaluated_at="2026-11-09T00:00:00",
        eligible=True,
        require_manual_approval=False,
        criteria={},
        reason="全基準クリア。自動昇格可能。",
    )


def _write_challenger(unified_dir: Path, name: str, mtime: float) -> None:
    path = unified_dir / f"{name}.joblib"
    path.write_bytes(b"model")
    os.utime(path, (mtime, mtime))


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    unified = d / "unified"
    unified.mkdir(parents=True)
    last_week = (datetime.now(timezone.utc) - timedelta(days=8)).timestamp()
    for name in _CHALLENGERS:
        _write_challenger(unified, name, last_week)
    return d


def _run_twice(models_dir: Path, shadows: list, mock_train, mock_promote):
    """週の学習を2回（1回目は学習で落ちる）流し、知らせのモックを返す。"""
    from src.orchestration.jobs.weekly import run_weekly_training

    with ExitStack() as stack:
        stack.enter_context(patch("config.settings.AUTO_PROMOTE_MODEL", True))
        stack.enter_context(
            patch(
                "src.prediction.promotion_gate.evaluate_promotion",
                return_value=_eligible_result(),
            )
        )
        stack.enter_context(patch("src.prediction.promotion_gate.save_promotion_result"))
        stack.enter_context(
            patch("src.prediction.shadow_evaluation.evaluate_shadow_models", side_effect=shadows)
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
        stack.enter_context(
            patch("src.utils.data_path_utils.get_models_dir", return_value=str(models_dir))
        )
        stack.enter_context(patch("src.prediction.prediction_pipeline.run_accuracy_check"))
        stack.enter_context(patch("src.reporting.discord.discord_utils.send_drift_alert"))
        stack.enter_context(
            patch("src.reporting.discord.discord_utils.send_weekly_training_completion")
        )
        stack.enter_context(patch("src.reporting.discord.discord_utils.send_promotion_result"))
        stack.enter_context(patch("src.reporting.discord.discord_utils.send_webhook_text"))
        mock_notify = stack.enter_context(
            patch("src.orchestration.jobs.weekly_training._notify_stage_failure")
        )
        with pytest.raises(MemoryError):
            run_weekly_training()
        run_weekly_training()
    return mock_notify


def _read_record() -> dict:
    from src.utils.data_path_utils import get_results_dir

    year, week, _ = datetime.now(timezone.utc).isocalendar()
    path = Path(get_results_dir()) / "accuracy" / "shadow" / f"{year}-W{week:02d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _no_comparison_notices(mock_notify) -> list:
    return [c for c in mock_notify.call_args_list if "比べられなかった" in c.args[0]]


def _train_failing(models_dir: Path, fail_on: str):
    """fail_on の学習で1回だけ落ち、学習した challenger は今の時刻で保存し直す。"""
    state = {"failed": False}

    def _train(model_type, model_name):
        if model_type == fail_on and not state["failed"]:
            state["failed"] = True
            raise MemoryError(f"{model_type} の学習でメモリ不足")
        _write_challenger(models_dir / "unified", model_name, time.time())

    return MagicMock(side_effect=_train)


def _promote_mock() -> MagicMock:
    return MagicMock(return_value={"promoted": _PRODUCTIONS, "skipped": []})


def _assert_first_run_kept(record: dict, first: dict) -> None:
    assert record["promoted"] is True
    assert record["gate_evaluated"] is True
    assert record["eligible"] is True
    assert record["gate_skipped_reason"] is None
    for key in (
        "production_hit_rate",
        "production_sharpe",
        "challenger_hit_rate",
        "challenger_sharpe",
        "n_production",
        "n_challenger",
        "window_days",
        "challenger_wins",
    ):
        assert record[key] == first[key], key
    assert record["retry_count"] == 1
    assert record["last_run_at"] is not None


def test_retry_after_promotion_and_xgboost_failure_keeps_first_record(models_dir):
    """1回目で昇格し XGBoost の学習で落ちたやり直しは、1回目の比べと昇格を残す。"""
    first = _compared(wins=True)
    mock_promote = _promote_mock()
    mock_notify = _run_twice(
        models_dir,
        [first, _empty_window()],
        _train_failing(models_dir, "XGBoostModel"),
        mock_promote,
    )

    _assert_first_run_kept(_read_record(), first)
    assert _no_comparison_notices(mock_notify) == []
    assert mock_promote.call_count == 1


def test_retry_after_promotion_and_lightgbm_failure_keeps_first_record(models_dir):
    """XGBoost だけ学習し直してから LightGBM で落ちたやり直しも、1回目の記録を残す。"""
    first = _compared(wins=True)
    mock_promote = _promote_mock()
    mock_notify = _run_twice(
        models_dir,
        [first, _empty_window()],
        _train_failing(models_dir, "LightGBMModel"),
        mock_promote,
    )

    _assert_first_run_kept(_read_record(), first)
    assert _no_comparison_notices(mock_notify) == []
    assert mock_promote.call_count == 1


def test_retry_after_loss_keeps_challenger_lost(models_dir):
    """負けの週に XGBoost だけ学習し直して落ちたやり直しは、負けの記録を残す。"""
    first = _compared(wins=False)
    mock_notify = _run_twice(
        models_dir,
        [first, _empty_window()],
        _train_failing(models_dir, "LightGBMModel"),
        _promote_mock(),
    )

    record = _read_record()
    assert record["gate_skipped_reason"] == "challenger_lost"
    assert record["challenger_hit_rate"] == 0.45
    assert record["promoted"] is False
    assert _no_comparison_notices(mock_notify) == []


def test_retry_after_no_comparison_still_records_no_shadow_data(models_dir):
    """前の回も比べられなかった週は、今までどおり no_shadow_data と書いて知らせる。"""
    mock_notify = _run_twice(
        models_dir,
        [_empty_window(), _empty_window()],
        _train_failing(models_dir, "LightGBMModel"),
        _promote_mock(),
    )

    record = _read_record()
    assert record["gate_skipped_reason"] == "no_shadow_data"
    assert record["promoted"] is False
    assert record["gate_evaluated"] is False
    assert "retry_count" not in record
    assert len(_no_comparison_notices(mock_notify)) == 2
