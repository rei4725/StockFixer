"""影の比べと昇格ゲートが、モデルごとの直近 N 日分の採点の行を全部読むことのテスト（#958）。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from src.utils.db._bulk import bulk_insert
from src.utils.db._connection import _db_connection

_N_SYMBOLS = 700
_N_DAYS = 20
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


def _days(n: int) -> list[str]:
    return [d.strftime("%Y%m%d_224500") for d in pd.bdate_range(end="2026-10-21", periods=n)]


def _rows(model_name: str, days: list[str], hit_days: int) -> pd.DataFrame:
    """days の各日に _N_SYMBOLS 行を作る。直近 hit_days 日だけ方向が当たり、誤差 0 にする。"""
    frames = []
    for i, day in enumerate(days):
        hit = i >= len(days) - hit_days
        frames.append(
            pd.DataFrame(
                {
                    "market": "jp",
                    "symbol": [f"{n:04d}" for n in range(_N_SYMBOLS)],
                    "model_name": model_name,
                    "predicted_at": day,
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


class TestShadowEvaluationReadsWholeWindow(unittest.TestCase):
    def setUp(self):
        days = _days(_N_DAYS)
        _insert(_rows("production", days, hit_days=10))
        _insert(_rows("challenger", days, hit_days=3))
        # 今の challenger は20日より前に学習した（版の境で絞らない。#1327）
        trained = patch(
            "src.prediction.challenger_window.current_challenger_trained_at",
            return_value=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
        trained.start()
        self.addCleanup(trained.stop)

    def test_counts_are_twenty_days_each(self):
        from src.prediction.shadow_evaluation import evaluate_shadow_models

        with patch("src.prediction.shadow_evaluation._record_ab_result"):
            result = evaluate_shadow_models(evaluation_days=_N_DAYS)

        self.assertEqual(result["n_production"], _N_DAYS * _N_SYMBOLS)
        self.assertEqual(result["n_challenger"], _N_DAYS * _N_SYMBOLS)
        self.assertAlmostEqual(result["production_hit_rate"], 10 / _N_DAYS)
        self.assertAlmostEqual(result["challenger_hit_rate"], 3 / _N_DAYS)

    def test_promotion_gate_uses_whole_window(self):
        from src.prediction.promotion_gate import _compute_hit_rate, _compute_slippage

        hit = _compute_hit_rate("challenger", horizon=1)
        slip = _compute_slippage("challenger", horizon=1)

        self.assertAlmostEqual(hit, 3 / _N_DAYS)
        self.assertAlmostEqual(slip, 0.04 * (_N_DAYS - 3) / _N_DAYS)


class TestLoadPredictionAccuracyWindow(unittest.TestCase):
    def test_reads_latest_days_of_one_model(self):
        from src.prediction.db import load_prediction_accuracy

        days = _days(5)
        _insert(_rows("production", days, hit_days=0))
        _insert(_rows("challenger", days[-1:], hit_days=0))

        df = load_prediction_accuracy(horizon=1, limit=0, model_name="production", days=3)

        self.assertEqual(len(df), 3 * _N_SYMBOLS)
        self.assertEqual(set(df["model_name"]), {"production"})
        self.assertEqual(set(df["predicted_at"]), set(days[-3:]))

    def test_no_rows_returns_empty(self):
        from src.prediction.db import load_prediction_accuracy

        df = load_prediction_accuracy(horizon=1, limit=0, model_name="challenger", days=3)

        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
