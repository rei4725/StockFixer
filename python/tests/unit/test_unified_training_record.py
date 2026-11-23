"""統合モデルの学習の確かめの数字を results/unified_training に残すことのテスト（#2131）。"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.prediction.constant_prediction import check_unified_training
from src.utils.data_path_utils import get_results_dir

_KEYS = {
    "model_name",
    "recorded_at",
    "best_iteration",
    "trees",
    "val_rows",
    "val_pred_std",
    "is_constant",
    "val_predicted_up_rate",
    "val_actual_up_rate",
}
# UTC ではまだ 11-23 の時刻。ファイル名は JST の日付になる
_NOW = datetime(2026, 11, 24, 0, 30, tzinfo=ZoneInfo("Asia/Tokyo"))


class _FakeModel:
    model_name = "unified_x"

    def __init__(self, predictions: list[float]) -> None:
        self._predictions = predictions
        self.predict_calls = 0
        self.model = SimpleNamespace(
            best_iteration=7,
            get_booster=lambda: SimpleNamespace(num_boosted_rounds=lambda: 20),
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self.predict_calls += 1
        return pd.Series(self._predictions, index=X.index)


def _eval_set(y: list[float]) -> list:
    return [(pd.DataFrame({"a": range(len(y))}), pd.Series(y))]


def _record_dir() -> str:
    return os.path.join(get_results_dir(), "unified_training")


def _read_record() -> dict:
    with open(os.path.join(_record_dir(), "2026-11-24_unified_x.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def _fixed_now():
    with patch("src.utils.japan_time.datetime") as mock_datetime:
        mock_datetime.now.side_effect = _NOW.astimezone
        yield


@pytest.fixture(autouse=True)
def notifier():
    port = MagicMock()
    with patch("src.prediction.ports._notification_port", port):
        yield port


def test_record_has_all_keys_and_up_rates(notifier):
    """検証の行がある回は鍵がそろい、上げの割合は 0 を含めずに数えた値になる。"""
    model = _FakeModel([0.01, 0.02, 0.03, 0.0, -0.01])

    check_unified_training(model, _eval_set([0.01, -0.02, 0.0, -0.01, 0.02]))

    record = _read_record()
    assert set(record) == _KEYS
    assert record["model_name"] == "unified_x"
    assert record["recorded_at"] == "2026-11-23T15:30:00+00:00"
    assert record["best_iteration"] == 7
    assert record["trees"] == 20
    assert record["val_rows"] == 5
    assert record["val_pred_std"] == pytest.approx(np.sqrt(0.0002))
    assert record["is_constant"] is False
    assert record["val_predicted_up_rate"] == pytest.approx(3 / 5)
    assert record["val_actual_up_rate"] == pytest.approx(2 / 5)
    assert model.predict_calls == 1
    notifier.send_alert.assert_not_called()


def test_record_without_validation_rows_has_nulls():
    """検証の行が無い回も書き、検証から出す値は null になる。"""
    check_unified_training(_FakeModel([]), None)

    record = _read_record()
    assert set(record) == _KEYS
    assert record["val_rows"] == 0
    assert record["val_pred_std"] is None
    assert record["is_constant"] is None
    assert record["val_predicted_up_rate"] is None
    assert record["val_actual_up_rate"] is None


def test_same_day_retraining_overwrites():
    """同じ日の同じモデルの学び直しは上書きし、定数の印も後の回の値になる。"""
    y = [0.01, -0.01]
    check_unified_training(_FakeModel([0.02, -0.02]), _eval_set(y))
    check_unified_training(_FakeModel([0.02, 0.02]), _eval_set(y))

    assert os.listdir(_record_dir()) == ["2026-11-24_unified_x.json"]
    record = _read_record()
    assert record["is_constant"] is True
    assert record["val_predicted_up_rate"] == 1.0
    assert record["val_actual_up_rate"] == 0.5


def _assert_log_and_notice_as_before(caplog, notifier) -> None:
    lines = [r for r in caplog.records if "統合モデルの学習の確かめ: " in r.getMessage()]
    assert [r.getMessage() for r in lines] == [
        "統合モデルの学習の確かめ: model=unified_x "
        "best_iteration=7 trees=20 val_rows=2 val_pred_std=0.0"
    ]
    notifier.send_alert.assert_called_once()
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1
    assert "学習の記録を書けず" in errors[0].getMessage()


def test_unwritable_target_keeps_log_and_notice(caplog, notifier):
    """書き込み先に書けない回も例外を投げず、ログの1行と定数の知らせは変わらない。"""
    os.makedirs(get_results_dir(), exist_ok=True)
    with open(_record_dir(), "w", encoding="utf-8") as f:
        f.write("フォルダの場所をファイルで塞ぐ")

    with caplog.at_level(logging.INFO):
        check_unified_training(_FakeModel([0.02, 0.02]), _eval_set([0.01, -0.01]))

    _assert_log_and_notice_as_before(caplog, notifier)


def test_unreadable_actual_values_keep_log_and_notice(caplog, notifier):
    """実際の値が数字にならず割合を出せない回も、ログの1行と定数の知らせは変わらない。"""
    eval_set = [(pd.DataFrame({"a": [0, 1]}), ["x", "y"])]

    with caplog.at_level(logging.INFO):
        check_unified_training(_FakeModel([0.02, 0.02]), eval_set)

    _assert_log_and_notice_as_before(caplog, notifier)
