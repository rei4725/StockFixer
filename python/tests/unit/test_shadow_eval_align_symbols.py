"""影の比べと昇格ゲートが、production を challenger と同じ日・market・symbol に絞るテスト（#1441）。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from src.utils.db._bulk import bulk_insert
from src.utils.db._connection import _db_connection

_N_SYMBOLS = 50
_N_MISSING = 10
_COLUMNS = [
    "market",
    "symbol",
    "model_name",
    "predicted_at",
    "horizon",
    "predicted_ratio",
    "actual_ratio",
    "direction_match",
]
_DAYS = list(pd.bdate_range(end="2026-10-21", periods=4))
_BOUNDARY = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


def _symbols(n: int) -> list[str]:
    return [f"{i:04d}" for i in range(n)]


def _rows(model_name: str, symbols: list[str]) -> pd.DataFrame:
    """各日に symbols の行を作る。末尾 _N_MISSING 銘柄（challenger が欠ける銘柄）は外れる。"""
    hard = set(_symbols(_N_SYMBOLS)[-_N_MISSING:])
    frames = []
    for day in _DAYS:
        actual = [-0.03 if s in hard else 0.01 + 0.001 * (i % 5) for i, s in enumerate(symbols)]
        frames.append(
            pd.DataFrame(
                {
                    "market": "jp",
                    "symbol": symbols,
                    "model_name": model_name,
                    "predicted_at": day.strftime("%Y%m%d_224500"),
                    "horizon": 1,
                    "predicted_ratio": 0.01,
                    "actual_ratio": actual,
                    "direction_match": [s not in hard for s in symbols],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _write_challenger(models_dir: str, mtime: datetime) -> None:
    unified = os.path.join(models_dir, "unified")
    os.makedirs(unified, exist_ok=True)
    for name in ("UnifiedStockXGBoost_challenger", "UnifiedStockLightGBM_challenger"):
        path = os.path.join(unified, f"{name}.joblib")
        with open(path, "wb") as f:
            f.write(b"x")
        os.utime(path, (mtime.timestamp(), mtime.timestamp()))


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._models = patch(
            "src.utils.data_path_utils.get_models_dir", return_value=self._tmp.name
        )
        self._models.start()
        self.addCleanup(self._models.stop)
        self.kept = _symbols(_N_SYMBOLS - _N_MISSING)
        self.chal = _rows("challenger", self.kept)
        self.prod = _rows("production", _symbols(_N_SYMBOLS))


class TestFilterToDaysAlignsSymbols(_Base):
    def test_drops_production_rows_of_symbols_missing_in_challenger(self):
        from src.prediction.challenger_window import filter_current_challenger, filter_to_days

        _write_challenger(self._tmp.name, _BOUNDARY)
        _, window = filter_current_challenger(self.chal)

        kept = filter_to_days(self.prod, window)

        self.assertEqual(window.n_production, window.n_challenger)
        self.assertEqual(window.n_production, len(_DAYS) * len(self.kept))
        self.assertEqual(sorted(set(kept["symbol"])), self.kept)

    def test_other_market_with_same_symbol_is_dropped(self):
        from src.prediction.challenger_window import filter_current_challenger, filter_to_days

        _write_challenger(self._tmp.name, _BOUNDARY)
        _, window = filter_current_challenger(self.chal)
        us = self.prod.assign(market="us")

        kept = filter_to_days(pd.concat([self.prod, us], ignore_index=True), window)

        self.assertEqual(set(kept["market"]), {"jp"})
        self.assertEqual(window.n_production, window.n_challenger)

    def test_no_challenger_file_keeps_empty(self):
        from src.prediction.challenger_window import filter_current_challenger, filter_to_days

        _, window = filter_current_challenger(self.chal)

        self.assertTrue(filter_to_days(self.prod, window).empty)
        self.assertEqual(window.n_production, 0)

    def test_zero_challenger_rows_keeps_empty(self):
        from src.prediction.challenger_window import filter_current_challenger, filter_to_days

        _write_challenger(self._tmp.name, _BOUNDARY)
        _, window = filter_current_challenger(self.chal.iloc[0:0])

        self.assertTrue(filter_to_days(self.prod, window).empty)
        self.assertEqual(window.n_production, 0)


class TestShadowEvaluationAlignsSymbols(_Base):
    def test_production_rows_of_missing_symbols_are_not_scored(self):
        from src.prediction.shadow_evaluation import _compute_sharpe, evaluate_shadow_models

        with _db_connection() as con:
            bulk_insert(con, "prediction_accuracy", self.chal, columns=_COLUMNS)
            bulk_insert(con, "prediction_accuracy", self.prod, columns=_COLUMNS)
        _write_challenger(self._tmp.name, _BOUNDARY)

        with patch("src.prediction.shadow_evaluation._record_ab_result"):
            result = evaluate_shadow_models(evaluation_days=len(_DAYS))

        expected = self.prod[self.prod["symbol"].isin(self.kept)]
        self.assertEqual(result["n_production"], result["n_challenger"])
        self.assertEqual(result["n_production"], len(expected))
        self.assertAlmostEqual(result["production_hit_rate"], 1.0)
        self.assertAlmostEqual(result["production_sharpe"], _compute_sharpe(expected))
        self.assertAlmostEqual(result["production_sharpe"], result["challenger_sharpe"])
        # 同じ組で同じ成績なので、challenger は上回らない
        self.assertFalse(result["challenger_wins"])


if __name__ == "__main__":
    unittest.main()
