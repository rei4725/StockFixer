"""実物の XGBoost・LightGBM・MLP で、学習の確かめの数字が読めることのテスト（#1602）。

フェイクでは属性の名前の違い（best_iteration と best_iteration_）を確かめられないため、
小さな表で実際に学習する。pytest-xdist の下では実物の学習が取り合いで不安定になるため
（#615）、並列の worker の中では飛ばし、直列で流したときだけ動かす。
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.prediction.constant_prediction import check_unified_training
from src.prediction.models.lightgbm import LightGBMModel
from src.prediction.models.transformer_model import TransformerModel
from src.prediction.models.xgboost import XGBoostModel

pytestmark = pytest.mark.skipif(
    "PYTEST_XDIST_WORKER" in os.environ, reason="実物の学習は xdist の下で不安定（#615）"
)


def _frames() -> tuple[pd.DataFrame, pd.Series, list]:
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = pd.Series(X["a"] * 0.02 + rng.normal(scale=0.002, size=300))
    return X.iloc[:240], y.iloc[:240], [(X.iloc[240:], y.iloc[240:])]


@pytest.mark.parametrize(
    "model",
    [
        XGBoostModel("X", n_estimators=20, n_jobs=1),
        LightGBMModel("L", n_estimators=20, n_jobs=1, min_child_samples=5),
    ],
    ids=["xgboost", "lightgbm"],
)
def test_tree_models_report_best_iteration_and_trees(model, caplog):
    X, y, eval_set = _frames()
    model.train(X, y, eval_set=eval_set)

    with (
        caplog.at_level(logging.INFO),
        patch("src.prediction.ports._notification_port", None),
    ):
        check_unified_training(model, eval_set)

    assert "val_rows=60" in caplog.text
    assert "best_iteration=None" not in caplog.text
    assert "trees=None" not in caplog.text
    assert "val_pred_std=None" not in caplog.text


def test_mlp_has_no_best_iteration_and_is_still_checked(caplog):
    """MLP は best_iteration も木の数も持たない。落とさず None と書く。"""
    model = TransformerModel("M", max_iter=5, hidden_layer_sizes=(4,), early_stopping=False)
    X, y, eval_set = _frames()
    model.train(X, y, eval_set=eval_set)

    with (
        caplog.at_level(logging.INFO),
        patch("src.prediction.ports._notification_port", None),
    ):
        check_unified_training(model, eval_set)

    assert "best_iteration=None" in caplog.text
    assert "trees=None" in caplog.text
    assert "val_pred_std=None" not in caplog.text
