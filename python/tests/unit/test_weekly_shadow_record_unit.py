"""週の学習の影の比べと昇格ゲートの結果を results/accuracy/shadow/ に書くことのテスト（#1503）。"""

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

_EXPECTED_KEYS = {
    "week",
    "written_at",
    "shadow_evaluated",
    "shadow_failed",
    "shadow_error",
    "challenger_trained_at",
    "production_changed_at",
    "days",
    "window_days",
    "production_hit_rate",
    "production_sharpe",
    "challenger_hit_rate",
    "challenger_sharpe",
    "n_production",
    "n_challenger",
    "challenger_wins",
    "fresh_challengers",
    "gate_evaluated",
    "gate_skipped_reason",
    "gate_failed",
    "eligible",
    "reason",
    "promoted",
}


def _shadow(wins: bool) -> dict:
    return {
        "production_hit_rate": 0.537,
        "production_sharpe": 0.4,
        "challenger_hit_rate": 0.6 if wins else 0.454,
        "challenger_sharpe": 0.8 if wins else 0.1,
        "challenger_wins": wins,
        "evaluated_at": "20261102_160000",
        "n_production": 3124,
        "n_challenger": 2342,
        "challenger_trained_at": "2026-10-26T16:30:00+00:00",
        "window_days": 3,
    }


def _eligible_result() -> PromotionResult:
    return PromotionResult(
        shadow_model="challenger",
        current_model="production",
        evaluated_at="2026-11-02T00:00:00",
        eligible=True,
        require_manual_approval=False,
        criteria={},
        reason="全基準クリア。自動昇格可能。",
    )


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    unified = d / "unified"
    unified.mkdir(parents=True)
    last_week = (datetime.now(timezone.utc) - timedelta(days=8)).timestamp()
    for name in _CHALLENGERS:
        path = unified / f"{name}.joblib"
        path.write_bytes(b"model")
        os.utime(path, (last_week, last_week))
    return d


def _run(stack: ExitStack, models_dir: Path, shadow, mock_train=None, mock_promote=None):
    from src.orchestration.jobs.weekly import run_weekly_training

    stack.enter_context(patch("config.settings.AUTO_PROMOTE_MODEL", True))
    stack.enter_context(
        patch("src.prediction.promotion_gate.evaluate_promotion", return_value=_eligible_result())
    )
    stack.enter_context(patch("src.prediction.promotion_gate.save_promotion_result"))
    if isinstance(shadow, Exception):
        shadow_patch = patch(
            "src.prediction.shadow_evaluation.evaluate_shadow_models", side_effect=shadow
        )
    else:
        shadow_patch = patch(
            "src.prediction.shadow_evaluation.evaluate_shadow_models", return_value=shadow
        )
    stack.enter_context(shadow_patch)
    stack.enter_context(
        patch(
            "src.prediction.shadow_evaluation.promote_unified_challenger",
            mock_promote or MagicMock(return_value={"promoted": _PRODUCTIONS, "skipped": []}),
        )
    )
    stack.enter_context(
        patch("src.prediction.shadow_evaluation._UNIFIED_PRODUCTION_NAMES", _PRODUCTIONS)
    )
    stack.enter_context(
        patch("src.prediction.shadow_evaluation._UNIFIED_CHALLENGER_NAMES", _CHALLENGERS)
    )
    stack.enter_context(
        patch(
            "src.prediction.unified_model_pipeline.train_unified_model",
            mock_train or MagicMock(),
        )
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
    run_weekly_training()


def _read_record() -> dict:
    from src.utils.data_path_utils import get_results_dir

    shadow_dir = Path(get_results_dir()) / "accuracy" / "shadow"
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    path = shadow_dir / f"{year}-W{week:02d}.json"
    assert path.exists(), f"{path} が無い"
    assert [p.name for p in shadow_dir.iterdir()] == [path.name]
    record = json.loads(path.read_text(encoding="utf-8"))
    assert set(record) == _EXPECTED_KEYS
    assert record["week"] == f"{year}-W{week:02d}"
    return record


def test_record_when_challenger_loses(models_dir):
    with ExitStack() as stack:
        _run(stack, models_dir, _shadow(wins=False))

    record = _read_record()
    assert record["shadow_evaluated"] is True
    assert record["shadow_failed"] is False
    assert record["challenger_wins"] is False
    assert record["production_hit_rate"] == 0.537
    assert record["challenger_hit_rate"] == 0.454
    assert record["n_production"] == 3124
    assert record["n_challenger"] == 2342
    assert record["window_days"] == 3
    assert record["days"] == 20
    assert record["challenger_trained_at"] == "2026-10-26T16:30:00+00:00"
    assert record["gate_evaluated"] is False
    assert record["gate_skipped_reason"] == "challenger_lost"
    assert record["eligible"] is None
    assert record["promoted"] is False


def test_record_when_challenger_wins_and_passes_gate(models_dir):
    with ExitStack() as stack:
        _run(stack, models_dir, _shadow(wins=True))

    record = _read_record()
    assert record["challenger_wins"] is True
    assert record["gate_evaluated"] is True
    assert record["gate_skipped_reason"] is None
    assert record["gate_failed"] is False
    assert record["eligible"] is True
    assert record["reason"] == "全基準クリア。自動昇格可能。"
    assert record["promoted"] is True


def test_record_when_challenger_wins_but_trained_this_week(models_dir):
    now = time.time()
    for name in _CHALLENGERS:
        os.utime(models_dir / "unified" / f"{name}.joblib", (now, now))
    mock_promote = MagicMock()
    with ExitStack() as stack:
        _run(stack, models_dir, _shadow(wins=True), mock_promote=mock_promote)

    mock_promote.assert_not_called()
    record = _read_record()
    assert record["challenger_wins"] is True
    assert record["gate_evaluated"] is False
    assert record["gate_skipped_reason"] == "challenger_trained_this_week"
    assert record["fresh_challengers"] == _CHALLENGERS
    assert record["promoted"] is False


def test_record_when_shadow_evaluation_fails(models_dir):
    with ExitStack() as stack:
        _run(stack, models_dir, RuntimeError("DB が読めない"))

    record = _read_record()
    assert record["shadow_evaluated"] is False
    assert record["shadow_failed"] is True
    assert "DB が読めない" in record["shadow_error"]
    assert record["challenger_wins"] is None
    assert record["gate_evaluated"] is False
    assert record["gate_skipped_reason"] == "shadow_failed"
    assert record["promoted"] is False


def test_write_failure_does_not_stop_weekly_training(models_dir):
    mock_train = MagicMock()
    with ExitStack() as stack:
        mock_notify = stack.enter_context(
            patch("src.orchestration.jobs.weekly_training._notify_stage_failure")
        )
        stack.enter_context(
            patch(
                "src.prediction.shadow_record.save_weekly_shadow_record",
                side_effect=OSError("ディスクがいっぱい"),
            )
        )
        _run(stack, models_dir, _shadow(wins=False), mock_train=mock_train)

    assert mock_train.call_count == 2
    labels = [c.args[0] for c in mock_notify.call_args_list]
    assert any("影の比べの記録" in label for label in labels)


def test_save_weekly_shadow_record_uses_iso_week_name(tmp_path):
    from src.prediction.shadow_record import save_weekly_shadow_record

    path = save_weekly_shadow_record(
        {"promoted": False},
        now=datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc),
        results_dir=str(tmp_path),
    )
    # 2027-01-01 は ISO 週では 2026 年の第53週
    assert Path(path) == tmp_path / "accuracy" / "shadow" / "2026-W53.json"
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    assert record["week"] == "2026-W53"
    assert record["promoted"] is False
