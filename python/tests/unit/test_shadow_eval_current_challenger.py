"""影の比べと昇格ゲートが、今の challenger の予想の日だけで比べることのテスト（#1327）。"""

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
_N_DAYS = 20
_N_NEW = 4
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
_DAYS = list(pd.bdate_range(end="2026-10-21", periods=_N_DAYS))


def _rows(model_name: str, hits: list[bool]) -> pd.DataFrame:
    """各日に _N_SYMBOLS 行を作る。hits[i] の日は方向が当たる。"""
    frames = []
    for day, hit in zip(_DAYS, hits):
        frames.append(
            pd.DataFrame(
                {
                    "market": "jp",
                    "symbol": [f"{n:04d}" for n in range(_N_SYMBOLS)],
                    "model_name": model_name,
                    "predicted_at": day.strftime("%Y%m%d_224500"),
                    "horizon": 1,
                    "predicted_ratio": 0.01,
                    "actual_ratio": 0.01 if hit else -0.03,
                    "direction_match": hit,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _insert(df: pd.DataFrame) -> None:
    with _db_connection() as con:
        bulk_insert(con, "prediction_accuracy", df, columns=_COLUMNS)


def _write_challenger(models_dir: str, mtime: datetime) -> None:
    unified = os.path.join(models_dir, "unified")
    os.makedirs(unified, exist_ok=True)
    for name in ("UnifiedStockXGBoost_challenger", "UnifiedStockLightGBM_challenger"):
        path = os.path.join(unified, f"{name}.joblib")
        with open(path, "wb") as f:
            f.write(b"x")
        os.utime(path, (mtime.timestamp(), mtime.timestamp()))


class TestCurrentChallengerWindow(unittest.TestCase):
    def setUp(self):
        # 前の版の16日は全部当たり、今の版の4日は1日だけ当たる
        old = [True] * (_N_DAYS - _N_NEW)
        _insert(_rows("challenger", old + [False, False, False, True]))
        # production は直近2日だけ当たる（20日なら 0.1、今の4日なら 0.5）
        _insert(_rows("production", [False] * (_N_DAYS - 2) + [True, True]))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._models = patch(
            "src.utils.data_path_utils.get_models_dir", return_value=self._tmp.name
        )
        self._models.start()
        self.addCleanup(self._models.stop)

    def _set_boundary(self, mtime: datetime) -> None:
        _write_challenger(self._tmp.name, mtime)

    def _shadow(self) -> dict:
        from src.prediction.shadow_evaluation import evaluate_shadow_models

        with patch("src.prediction.shadow_evaluation._record_ab_result"):
            return evaluate_shadow_models(evaluation_days=_N_DAYS)

    def _gate(self):
        from src.prediction.promotion_gate import evaluate_promotion

        wf = pd.DataFrame({"total_return": [0.05], "max_drawdown": [-0.1], "sharpe_ratio": [0.5]})
        with (
            patch(
                "src.prediction.promotion_gate._load_latest_wf_summary",
                return_value=(wf, "wf_summary_20261021_160001.csv"),
            ),
            patch("src.prediction.promotion_gate._collect_run_ids", return_value=[]),
        ):
            return evaluate_promotion("challenger", "production")

    def _boundary_before_new_days(self) -> datetime:
        last_old = _DAYS[_N_DAYS - _N_NEW - 1]
        return datetime(last_old.year, last_old.month, last_old.day, 23, 30, tzinfo=timezone.utc)

    def test_shadow_uses_only_current_challenger_days(self):
        self._set_boundary(self._boundary_before_new_days())

        result = self._shadow()

        self.assertAlmostEqual(result["challenger_hit_rate"], 1 / _N_NEW)
        self.assertAlmostEqual(result["production_hit_rate"], 2 / _N_NEW)
        self.assertEqual(result["n_challenger"], _N_NEW * _N_SYMBOLS)
        self.assertEqual(result["n_production"], _N_NEW * _N_SYMBOLS)
        self.assertEqual(result["window_days"], _N_NEW)
        self.assertFalse(result["challenger_wins"])

    def test_gate_uses_only_current_challenger_days(self):
        self._set_boundary(self._boundary_before_new_days())

        result = self._gate()

        hit = result.criteria["hit_rate"]
        self.assertAlmostEqual(hit["shadow"], 1 / _N_NEW)
        self.assertAlmostEqual(hit["current"], 2 / _N_NEW)
        self.assertFalse(hit["passed"])
        self.assertEqual(hit["window_days"], _N_NEW)
        self.assertEqual(hit["window_n_challenger"], _N_NEW * _N_SYMBOLS)
        self.assertEqual(hit["window_n_production"], _N_NEW * _N_SYMBOLS)
        # 誤差は今の4日（当たり1日・外れ3日）の値
        self.assertAlmostEqual(result.criteria["slippage"]["shadow"], 0.04 * 3 / _N_NEW)
        self.assertEqual(result.baseline_metrics["current"]["n_baseline"], _N_NEW * _N_SYMBOLS)

    def test_no_challenger_file_means_no_data(self):
        result = self._shadow()
        self.assertFalse(result["challenger_wins"])
        self.assertIsNone(result["challenger_hit_rate"])
        self.assertEqual(result["n_challenger"], 0)

        gate = self._gate()
        self.assertIsNone(gate.criteria["hit_rate"]["shadow"])
        self.assertFalse(gate.criteria["hit_rate"]["passed"])

    def test_no_rows_after_boundary_means_no_data(self):
        self._set_boundary(datetime(2026, 10, 22, 0, 0, tzinfo=timezone.utc))

        result = self._shadow()
        self.assertFalse(result["challenger_wins"])
        self.assertIsNone(result["challenger_hit_rate"])
        self.assertEqual(result["window_days"], 0)

        gate = self._gate()
        self.assertIsNone(gate.criteria["hit_rate"]["shadow"])
        self.assertFalse(gate.criteria["hit_rate"]["passed"])


if __name__ == "__main__":
    unittest.main()
