"""ドリフトのテストが本物の models を読まないことのテスト（#1408）。

本番の作業場の models/ に、テストと同じ銘柄名のフォルダが7日以内に更新されてできても、
ドリフトの再学習を期待するテストが落ちないことを確かめる（#1402 の再発防止）。
本物の models/ には書かず、python のルートを一時フォルダに差し替えて「本物の models」を作る。
"""

import importlib

import pytest
from tests.unit.models_dir_isolation import isolate_models_dir

from src.orchestration.jobs.drift_check import _SYMBOL_MODEL_NAMES

# run_daily_drift_check() を呼び、get_models_dir を自分で patch しないテストのファイル（#1408）
_DRIFT_TEST_MODULES = (
    "tests.unit.test_drift_check_relative_mae",
    "tests.unit.test_drift_check_relative_hit_rate",
    "tests.unit.test_alert_streak_wiring",
    "tests.unit.test_db_prediction",
    "tests.unit.test_system_config",
    "tests.unit.test_drift_check_load_fail",
    "tests.unit.test_drift_check_drop_stale",
)


@pytest.mark.parametrize("module_name", _DRIFT_TEST_MODULES)
def test_drift_test_module_isolates_models_dir(module_name):
    """ドリフトのテストのファイルは、models を一時フォルダに向ける fixture を当てている。"""
    module = importlib.import_module(module_name)
    assert getattr(module, "isolate_models_dir", None) is isolate_models_dir


def test_degraded_symbol_retrained_despite_fresh_real_models(monkeypatch, tmp_path):
    """本物の models に jp_BAD が更新したてであっても、BAD の再学習を期待するテストが通る。"""
    real_root = tmp_path / "real_python_root"
    monkeypatch.setattr("src.utils.data_path_utils._PYTHON_ROOT", str(real_root))
    subdir = real_root / "models" / "jp_BAD"
    subdir.mkdir(parents=True)
    for name in _SYMBOL_MODEL_NAMES:
        (subdir / f"{name}.joblib").write_bytes(b"")

    mae = importlib.import_module("tests.unit.test_drift_check_relative_mae")
    hit_rate = importlib.import_module("tests.unit.test_drift_check_relative_hit_rate")
    mae.test_only_degraded_symbol_triggers()
    hit_rate.test_only_degraded_hit_rate_symbol_triggers()
