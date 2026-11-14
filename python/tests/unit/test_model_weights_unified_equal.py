"""統合モデルの重みが今の本番の DB の名前で等分になることを確かめる（#1806）。

本番の prediction_accuracy の model_name は production・challenger・unified だけ、
model_metrics は StockXGBoostModel・StockLightGBMModel だけである。統合モデルの
名前（UnifiedStock*）はどちらの表にも無いので、重みは等分になる。
"""

import logging
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from src.prediction.db import model_metrics

_MOD = "src.prediction.db.model_metrics"

# 本番の DB にある名前（2026-11-15 の GROUP BY model_name）
_ACCURACY_NAMES = {"production", "challenger", "unified"}
_METRICS_NAMES = {"StockXGBoostModel": 0.62, "StockLightGBMModel": 0.55}


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConnection:
    """本番の表にある名前の行だけを返す接続。"""

    def __init__(self):
        self.queries = []

    def execute(self, sql, params):
        self.queries.append((sql, params))
        name = params[2]
        if "FROM prediction_accuracy" in sql:
            rows = [(0.01, 0.02)] * 5 if name in _ACCURACY_NAMES else []
            return _FakeCursor(rows)
        if "FROM model_metrics" in sql:
            acc = _METRICS_NAMES.get(name)
            return _FakeCursor([(acc,)] if acc is not None else [])
        raise AssertionError(f"想定外の問い合わせ: {sql}")


@pytest.fixture(autouse=True)
def _reset_logged():
    model_metrics._EQUAL_WEIGHTS_LOGGED.clear()
    yield
    model_metrics._EQUAL_WEIGHTS_LOGGED.clear()


def _weights(con, names):
    with patch(f"{_MOD}._db_connection", return_value=nullcontext(con)):
        return model_metrics.load_model_weights("jp", "7203", names)


@pytest.mark.parametrize(
    "names",
    [
        ["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
        ["UnifiedStockXGBoost_5d", "UnifiedStockLightGBM_5d"],
        ["UnifiedStockXGBoost"],
    ],
)
def test_unified_names_are_not_in_production_tables(names):
    """前提: 統合モデルの名前は本番の2つの表のどちらにも無い。"""
    for name in names:
        assert name not in _ACCURACY_NAMES
        assert name not in _METRICS_NAMES


@pytest.mark.parametrize(
    "names",
    [
        ["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
        ["UnifiedStockXGBoost_3d", "UnifiedStockLightGBM_3d"],
        ["UnifiedStockXGBoost_10d", "UnifiedStockLightGBM_10d"],
    ],
)
def test_unified_weights_are_equal_without_queries(names):
    """統合モデルは等分で、当たらない問い合わせを流さない。"""
    con = _FakeConnection()
    assert _weights(con, names) == [0.5, 0.5]
    assert con.queries == []


def test_unified_single_model_weight_is_one():
    con = _FakeConnection()
    assert _weights(con, ["UnifiedStockLightGBM"]) == [1.0]
    assert con.queries == []


def test_unified_equal_weights_logged_once(caplog):
    """等分に落ちたことを INFO で1回だけ出す（銘柄ごとには出さない）。"""
    names = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
    with caplog.at_level(logging.INFO, logger=_MOD):
        for symbol in ("7203", "6758", "9984"):
            with patch(f"{_MOD}._db_connection", return_value=nullcontext(_FakeConnection())):
                assert model_metrics.load_model_weights("jp", symbol, names) == [0.5, 0.5]
    messages = [r for r in caplog.records if "等分" in r.getMessage()]
    assert len(messages) == 1
    assert messages[0].levelno == logging.INFO
    assert "UnifiedStockXGBoost" in messages[0].getMessage()


def test_per_symbol_models_still_use_model_metrics():
    """銘柄ごとのモデル（StockXGBoostModel 等）は今までどおり model_metrics から重みを引く。"""
    con = _FakeConnection()
    weights = _weights(con, ["StockXGBoostModel", "StockLightGBMModel"])
    assert len(con.queries) == 4
    assert weights[0] > weights[1]
    assert sum(weights) == pytest.approx(1.0)
