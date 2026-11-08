"""backtest_promotion.old_production（見張りの5日の旧 production の物差し）のテスト。"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.prediction.backtest_promotion import old_production as op

_BACKUPS = [
    "/models/unified/UnifiedStockXGBoost_20261001_223000.joblib",
    "/models/unified/UnifiedStockLightGBM_20261001_223000.joblib",
]
_SYMBOLS = ["0000", "0001", "0002", "0003"]


def _watch_rows(new: list[float], actual: list[float]) -> pd.DataFrame:
    """2026-10-02〜06 の UTC 22:30（JST の翌朝 07:30）に予想した jp の採点済みの行。"""
    rows = []
    for i in range(5):
        day = (pd.Timestamp("2026-10-02") + pd.Timedelta(days=i)).strftime("%Y%m%d")
        for symbol, p, a in zip(_SYMBOLS, new, actual):
            rows.append(
                {
                    "market": "jp",
                    "symbol": symbol,
                    "predicted_at": f"{day}_223000",
                    "predicted_ratio": p,
                    "actual_ratio": a,
                }
            )
    return pd.DataFrame(rows)


def _features(market: str, symbol: str, raise_on_error: bool = False) -> pd.DataFrame:
    days = pd.date_range("2026-09-28", "2026-10-08")
    return pd.DataFrame({"date": days, "f": [d.day for d in days], "y": 0.0, "market_encoded": 1})


def _ohlcv(market: str, symbol: str, start_date=None) -> pd.DataFrame:
    days = pd.date_range("2026-09-28", "2026-10-08")
    return pd.DataFrame({"Close": 100.0}, index=days)


@pytest.fixture
def env(monkeypatch):
    load = MagicMock(side_effect=lambda name, model_dir=None: f"model:{name}")
    monkeypatch.setattr(op, "load_unified_model", load)
    monkeypatch.setattr(op, "load_stock_features", MagicMock(side_effect=_features))
    monkeypatch.setattr(op, "load_raw_ohlcv", MagicMock(side_effect=_ohlcv))
    return {"load": load}


def _predict_with(old_by_symbol: dict, calls: list):
    def fake(latest_X, price, market, symbol, horizon, models, weights_fn=None):
        calls.append((int(latest_X["f"].iloc[0]), price, [name for name, _ in models]))
        value = old_by_symbol[symbol]
        if isinstance(value, Exception):
            raise value
        return None if value is None else SimpleNamespace(diff_ratio=value, model_count=2)

    return fake


def test_measures_old_production_on_the_same_rows(env, monkeypatch):
    calls: list = []
    old = {"0000": 0.03, "0001": 0.01, "0002": -0.01, "0003": -0.03}
    monkeypatch.setattr(op, "ensemble_diff_ratio", _predict_with(old, calls))
    # 新しいモデルは全部上がると言い順位も逆。旧 production は順位も向きも当たる
    rows = _watch_rows([0.01, 0.02, 0.03, 0.04], [0.02, 0.01, -0.01, -0.02])

    result = op.measure_old_production(_BACKUPS, rows)

    assert result["reason"] is None
    assert result["n_watch_rows"] == 20
    assert result["n_compared"] == 20
    assert result["old"]["n_baseline"] == 20
    assert result["new"]["n_baseline"] == 20
    assert result["old"]["rank_corr"] > 0 > result["new"]["rank_corr"]
    assert result["old"]["hit_rate_minus_majority"] > result["new"]["hit_rate_minus_majority"]
    # 写しは本番の名前で重みを引き、写しの置き場から読む
    assert {c[2][0] for c in calls} == {"UnifiedStockXGBoost"}
    assert all(c[2] == ["UnifiedStockXGBoost", "UnifiedStockLightGBM"] for c in calls)
    env["load"].assert_any_call("UnifiedStockXGBoost_20261001_223000", model_dir="/models/unified")
    # UTC 10-02 22:30 は JST 10-03 07:30。基準の日は引けていた 10-02 の行と終値
    assert sorted({c[0] for c in calls}) == [2, 3, 4, 5, 6]
    assert {c[1] for c in calls} == {100.0}


def test_unreadable_backup_is_not_measured(env):
    env["load"].side_effect = FileNotFoundError("写しが無い")

    result = op.measure_old_production(_BACKUPS, _watch_rows([0.01] * 4, [0.01] * 4))

    assert result["old"] is None and result["new"] is None
    assert result["n_compared"] == 0
    assert result["reason"].startswith(op.UNREADABLE_BACKUPS)
    assert "写しが無い" in result["reason"]


def test_real_loader_on_a_missing_file_is_not_measured(tmp_path, monkeypatch):
    from src.prediction.unified_model_pipeline import load_unified_model

    monkeypatch.setattr(op, "load_unified_model", load_unified_model)
    backups = [str(tmp_path / "UnifiedStockXGBoost_20261001_223000.joblib")]

    result = op.measure_old_production(backups, _watch_rows([0.01] * 4, [0.01] * 4))

    assert result["reason"].startswith(op.UNREADABLE_BACKUPS)


def test_no_backups_or_a_missing_model_is_not_measured(env):
    rows = _watch_rows([0.01] * 4, [0.01] * 4)
    assert op.measure_old_production([], rows)["reason"] == op.NO_BACKUPS
    result = op.measure_old_production(_BACKUPS[:1], rows)
    assert result["reason"].startswith(op.MISSING_BACKUPS)
    assert "UnifiedStockLightGBM" in result["reason"]


def test_no_prediction_on_the_same_rows_is_not_measured(env, monkeypatch):
    failed = {s: RuntimeError("全モデルが失敗") for s in _SYMBOLS}
    monkeypatch.setattr(op, "ensemble_diff_ratio", _predict_with(failed, []))

    result = op.measure_old_production(_BACKUPS, _watch_rows([0.01] * 4, [0.01] * 4))

    assert result["reason"] == op.NO_PREDICTIONS
    assert result["n_compared"] == 0
    assert result["skipped"]["no_prediction"] == 20
    assert result["old"] is None


def test_too_few_predictions_on_the_same_rows_is_not_measured(env, monkeypatch):
    # 4 銘柄のうち 1 銘柄しか予想が出ない（5 行 < 20 行の 8 割）
    old = {"0000": 0.01, "0001": None, "0002": None, "0003": None}
    monkeypatch.setattr(op, "ensemble_diff_ratio", _predict_with(old, []))

    result = op.measure_old_production(_BACKUPS, _watch_rows([0.01] * 4, [0.01] * 4))

    assert result["reason"].startswith(op.SHORT_PREDICTIONS)
    assert result["n_compared"] == 5
    assert result["need_rows"] == pytest.approx(16.0)
    assert result["old"] is None


def test_rows_without_features_or_price_are_skipped(env, monkeypatch):
    old = {s: 0.01 for s in _SYMBOLS}
    monkeypatch.setattr(op, "ensemble_diff_ratio", _predict_with(old, []))
    monkeypatch.setattr(op, "load_stock_features", MagicMock(return_value=pd.DataFrame()))

    result = op.measure_old_production(_BACKUPS, _watch_rows([0.01] * 4, [0.01] * 4))

    assert result["skipped"]["no_features"] == 20
    assert result["reason"] == op.NO_PREDICTIONS

    monkeypatch.setattr(op, "load_stock_features", MagicMock(side_effect=_features))
    monkeypatch.setattr(op, "load_raw_ohlcv", MagicMock(return_value=pd.DataFrame()))
    result = op.measure_old_production(_BACKUPS, _watch_rows([0.01] * 4, [0.01] * 4))
    assert result["skipped"]["no_price"] == 20
