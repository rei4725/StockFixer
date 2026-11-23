"""unit テストが本物の models を読まないことのテスト（#2133）。

本番の入れ替えで models/unified に読めないモデルと今の日時の写しができても、
自動デプロイの前のテストで落ちた6件が通ることを確かめる。
本物の models/ には書かず、python のルートを一時フォルダに差し替えて「本物の models」を作る。
"""

import importlib
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.utils.data_path_utils import get_models_dir

_PIPELINE = ("test_scheduler_pipeline_unit", "TestRunDailyPipeline")
_WINDOW = ("test_shadow_eval_model_window", "TestShadowEvaluationReadsWholeWindow")
_EVALUATION = ("test_shadow_evaluation_unit", "TestEvaluateShadowModels")

# 2026-11-24 01:47 JST の自動デプロイで落ちた6件
_FAILED_TESTS = (
    (*_PIPELINE, "test_daily_pipeline_runs_all_steps"),
    (*_PIPELINE, "test_accuracy_check_failure_does_not_stop_pipeline"),
    (*_PIPELINE, "test_data_step_half_skipped_continues"),
    (*_WINDOW, "test_counts_are_twenty_days_each"),
    (*_WINDOW, "test_promotion_gate_uses_whole_window"),
    (*_EVALUATION, "test_challenger_wins_when_both_better"),
)


@pytest.fixture
def replaced_real_models(monkeypatch, tmp_path):
    """読めない本番のモデルと、今の日時の写しがある「本物の models」を作る。"""
    real_root = tmp_path / "real_python_root"
    unified = real_root / "models" / "unified"
    unified.mkdir(parents=True)
    stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    (unified / "UnifiedStockXGBoost.joblib").write_bytes(b"broken model")
    (unified / f"UnifiedStockXGBoost_{stamp}.joblib").write_bytes(b"broken model")
    monkeypatch.setattr("src.utils.data_path_utils._PYTHON_ROOT", str(real_root))
    return real_root / "models"


def test_models_dir_is_not_the_real_one(replaced_real_models):
    """本物の models があっても、get_models_dir は一時フォルダを指す。"""
    assert get_models_dir() != str(replaced_real_models)


@pytest.mark.parametrize(("module_name", "class_name", "method"), _FAILED_TESTS)
def test_passes_despite_replaced_real_models(replaced_real_models, module_name, class_name, method):
    """本物の models が入れ替わっていても、デプロイの前に落ちたテストが通る。"""
    module = importlib.import_module(f"tests.unit.{module_name}")
    test_class = getattr(module, class_name)
    if not issubclass(test_class, unittest.TestCase):
        getattr(test_class(), method)()
        return
    result = unittest.TestResult()
    test_class(method).run(result)
    problems = [trace for _, trace in result.errors + result.failures]
    assert not problems, "\n".join(problems)
    assert result.testsRun == 1
    assert not result.skipped
