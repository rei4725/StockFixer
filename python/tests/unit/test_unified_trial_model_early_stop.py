"""challenger と夜間の候補の学習で、early stopping が木を0〜数本で止めないことのテスト（#1774）。

日付で切った検証は、学習の期間と平均がずれ、雑音の多い目的変数では最初の木の後の
どの木も検証の誤差を増やす。early stopping はそこで止め、予想がほぼ定数になる。
小さな合成の表で実際に学習する。xdist の下では実物の学習が不安定なため飛ばす（#615）。
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.models.xgboost import XGBoostModel
from src.prediction.unified_model_pipeline import train_unified_model

pytestmark = pytest.mark.skipif(
    "PYTEST_XDIST_WORKER" in os.environ, reason="実物の学習は xdist の下で不安定（#615）"
)


def _noisy_table() -> pd.DataFrame:
    """雑音だけの目的変数で、直近の期間だけ平均がずれた表。"""
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2025-01-01", periods=300)
    n_sym = 10
    n = len(days) * n_sym
    df = pd.DataFrame(
        {
            "date": np.repeat(days, n_sym),
            "market": "us",
            "symbol": np.tile([f"S{i}" for i in range(n_sym)], len(days)),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    y = rng.normal(scale=0.01, size=n)
    y[df["date"] >= days[240]] += 0.005
    df["y"] = y
    return df


def _real_xgboost_model(model_name: str, monkeypatch) -> XGBoostModel:
    """本物の XGBoostModel を作る。

    test_model_manager は収集の時に sys.modules のこのモジュールを MagicMock に差し替える。
    その後に集めても本物で学習するよう、この間だけ読み直す。
    """
    name = "src.prediction.models.xgboost"
    monkeypatch.delitem(sys.modules, name, raising=False)
    real = importlib.import_module(name)
    return real.XGBoostModel(model_name, n_estimators=100, n_jobs=1)


def _train(model_name: str, tmp_path, monkeypatch) -> XGBoostModel:
    model = _real_xgboost_model(model_name, monkeypatch)
    manager = MagicMock()
    manager.create_model.return_value = model
    with (
        patch(
            "src.prediction.unified_model_pipeline.load_all_stock_data",
            return_value=_noisy_table(),
        ),
        patch("src.prediction.manager.ModelManager", return_value=manager),
        patch("src.prediction.ports._notification_port", None),
    ):
        train_unified_model(
            "XGBoostModel",
            model_name,
            save_dir=str(tmp_path),
            apply_saved_exclusions=False,
            save_selection=False,
        )
    return model


@pytest.mark.parametrize(
    "name", ["UnifiedStockXGBoost_challenger", "UnifiedStockXGBoost_candidate"]
)
def test_trial_model_uses_all_trees(name, tmp_path, monkeypatch):
    model = _train(name, tmp_path, monkeypatch)
    booster = model.model.get_booster()
    assert booster.num_boosted_rounds() == 100
    assert getattr(model.model, "best_iteration", None) in (None, 99)


def test_production_model_keeps_early_stopping(tmp_path, monkeypatch):
    """本番の名前の学習は今までどおり early stopping を使う。"""
    model = _train("UnifiedStockXGBoost", tmp_path, monkeypatch)
    assert model.model.best_iteration <= 5
