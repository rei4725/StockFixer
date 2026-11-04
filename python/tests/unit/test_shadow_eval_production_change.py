"""窓の途中で production が変わったら、変わり目より後の行だけで比べることのテスト（#1559）。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from src.prediction.backtest_promotion.state import copy_with_backup
from src.utils.db._bulk import bulk_insert
from src.utils.db._connection import _db_connection

_N_SYMBOLS = 50
_N_DAYS = 20
_N_CHALLENGER_DAYS = 8
_N_AFTER_CHANGE = 3
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
_OLD_MTIME = datetime(2026, 1, 5, tzinfo=timezone.utc)
_JST = ZoneInfo("Asia/Tokyo")


def _rows(model_name: str, hits: list[bool], time_of_day: str = "224500") -> pd.DataFrame:
    """直近の len(hits) 日に _N_SYMBOLS 行ずつ作る。hits[i] の日は方向が当たる。"""
    frames = []
    for day, hit in zip(_DAYS[-len(hits) :], hits):
        frames.append(
            pd.DataFrame(
                {
                    "market": "jp",
                    "symbol": [f"{n:04d}" for n in range(_N_SYMBOLS)],
                    "model_name": model_name,
                    "predicted_at": day.strftime(f"%Y%m%d_{time_of_day}"),
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


def _after_day(index: int) -> datetime:
    """_DAYS[index] の予想（22:45 UTC）より後で、次の日の予想より前の時刻。"""
    day = _DAYS[index]
    return datetime(day.year, day.month, day.day, 23, 30, tzinfo=timezone.utc)


def _touch(path: str, mtime: datetime) -> None:
    with open(path, "wb") as f:
        f.write(b"x")
    os.utime(path, (mtime.timestamp(), mtime.timestamp()))


def _write_challenger(models_dir: str, mtime: datetime) -> None:
    unified = os.path.join(models_dir, "unified")
    os.makedirs(unified, exist_ok=True)
    for name in ("UnifiedStockXGBoost_challenger", "UnifiedStockLightGBM_challenger"):
        _touch(os.path.join(unified, f"{name}.joblib"), mtime)


def _write_production(models_dir: str, mtime: datetime = _OLD_MTIME) -> None:
    unified = os.path.join(models_dir, "unified")
    os.makedirs(unified, exist_ok=True)
    for name in ("UnifiedStockXGBoost", "UnifiedStockLightGBM"):
        _touch(os.path.join(unified, f"{name}.joblib"), mtime)


def _change_production(models_dir: str, when: datetime) -> None:
    """入れ替え・戻し・昇格と同じ copy_with_backup で production のファイルを置き換える。

    置き換えた後のファイルの更新時刻は写し元のまま（古い）なので、ファイルの時刻では分からない。
    """
    unified = os.path.join(models_dir, "unified")
    pairs = []
    for name in ("UnifiedStockXGBoost", "UnifiedStockLightGBM"):
        src = os.path.join(unified, f"{name}_candidate.joblib")
        _touch(src, _OLD_MTIME)
        pairs.append((src, os.path.join(unified, f"{name}.joblib")))
    copy_with_backup(pairs, when.astimezone(_JST).strftime("%Y%m%d_%H%M%S"))


class _ModelsDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.models = self._tmp.name
        patcher = patch("src.utils.data_path_utils.get_models_dir", return_value=self.models)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestCurrentProductionChangedAt(_ModelsDirTestCase):
    def _changed_at(self) -> Optional[datetime]:
        from src.prediction.challenger_window import current_production_changed_at

        return current_production_changed_at()

    def test_none_without_models(self):
        self.assertIsNone(self._changed_at())

    def test_reads_the_jst_stamp_of_the_backup(self):
        _write_production(self.models)
        when = datetime(2026, 11, 4, 13, 20, 5, tzinfo=timezone.utc)

        _change_production(self.models, when)

        self.assertEqual(self._changed_at(), when)

    def test_latest_change_wins_including_numbered_backup(self):
        _write_production(self.models)
        first = datetime(2026, 11, 4, 13, 0, tzinfo=timezone.utc)
        second = datetime(2026, 11, 10, 13, 0, tzinfo=timezone.utc)
        _change_production(self.models, first)
        _change_production(self.models, second)
        _change_production(self.models, second)

        self.assertEqual(self._changed_at(), second)

    def test_ignores_other_models_and_unreadable_stamps(self):
        _write_production(self.models)
        unified = Path(self.models) / "unified"
        for name in (
            "UnifiedStockXGBoost_challenger_20261110_220000.joblib",
            "UnifiedStockXGBoost_candidate_20261110_220000.joblib",
            "UnifiedStockXGBoost_20269999_220000.joblib",
        ):
            (unified / name).write_bytes(b"x")

        self.assertEqual(self._changed_at(), _OLD_MTIME)

    def test_direct_rewrite_of_production_counts_as_a_change(self):
        """学習し直しのように写しを作らず本番のファイルを書いたときは、ファイルの時刻で分かる。"""
        when = datetime(2026, 11, 4, 13, 0, tzinfo=timezone.utc)
        _write_production(self.models, mtime=when)

        self.assertEqual(self._changed_at(), when)


class TestProductionChangeWindow(_ModelsDirTestCase):
    def setUp(self):
        super().setUp()
        # challenger は直近8日の版。その中の最後の3日の前に production が変わる
        n_old = _N_CHALLENGER_DAYS - _N_AFTER_CHANGE
        self.chal_hits = [True] * n_old + [False, False, True]
        self.prod_hits = [False] * (_N_DAYS - _N_AFTER_CHANGE) + [True, True, False]
        self.change = _after_day(_N_DAYS - _N_AFTER_CHANGE - 1)
        _insert(_rows("challenger", [True] * (_N_DAYS - _N_CHALLENGER_DAYS) + self.chal_hits))
        _write_challenger(self.models, _after_day(_N_DAYS - _N_CHALLENGER_DAYS - 1))
        _write_production(self.models)

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

    def test_shadow_uses_only_rows_after_the_change(self):
        from src.prediction.shadow_evaluation import _compute_sharpe

        _insert(_rows("production", self.prod_hits))
        _change_production(self.models, self.change)

        result = self._shadow()

        n_rows = _N_AFTER_CHANGE * _N_SYMBOLS
        self.assertAlmostEqual(result["challenger_hit_rate"], 1 / _N_AFTER_CHANGE)
        self.assertAlmostEqual(result["production_hit_rate"], 2 / _N_AFTER_CHANGE)
        self.assertAlmostEqual(
            result["challenger_sharpe"],
            _compute_sharpe(_rows("challenger", self.chal_hits[-_N_AFTER_CHANGE:])),
        )
        self.assertAlmostEqual(
            result["production_sharpe"],
            _compute_sharpe(_rows("production", self.prod_hits[-_N_AFTER_CHANGE:])),
        )
        self.assertEqual(result["n_challenger"], n_rows)
        self.assertEqual(result["n_production"], n_rows)
        self.assertEqual(result["window_days"], _N_AFTER_CHANGE)
        self.assertFalse(result["challenger_wins"])
        self.assertEqual(result["production_changed_at"], self.change.isoformat())

    def test_gate_uses_only_rows_after_the_change(self):
        _insert(_rows("production", self.prod_hits))
        _change_production(self.models, self.change)

        result = self._gate()

        n_rows = _N_AFTER_CHANGE * _N_SYMBOLS
        hit = result.criteria["hit_rate"]
        self.assertAlmostEqual(hit["shadow"], 1 / _N_AFTER_CHANGE)
        self.assertAlmostEqual(hit["current"], 2 / _N_AFTER_CHANGE)
        self.assertEqual(hit["window_days"], _N_AFTER_CHANGE)
        self.assertEqual(hit["window_n_challenger"], n_rows)
        self.assertEqual(hit["window_n_production"], n_rows)
        self.assertEqual(hit["production_changed_at"], self.change.isoformat())
        # 誤差は最後の3日（当たり1日・外れ2日）の値
        self.assertAlmostEqual(result.criteria["slippage"]["shadow"], 0.04 * 2 / _N_AFTER_CHANGE)
        self.assertEqual(result.baseline_metrics["current"]["n_baseline"], n_rows)
        self.assertEqual(result.baseline_metrics["shadow"]["n_baseline"], n_rows)
        self.assertFalse(result.eligible)

    def test_no_rows_after_the_change_means_challenger_does_not_win(self):
        """週次の直前に変わると、production の行は全部前のモデルで、比べる相手がいない。"""
        _insert(_rows("production", [False] * _N_DAYS))
        _change_production(self.models, _after_day(_N_DAYS - 1))

        result = self._shadow()

        self.assertFalse(result["challenger_wins"])
        self.assertIsNone(result["production_hit_rate"])
        self.assertIsNone(result["challenger_hit_rate"])
        self.assertEqual(result["n_production"], 0)
        self.assertEqual(result["n_challenger"], 0)
        self.assertEqual(result["window_days"], 0)

        gate = self._gate()
        self.assertIsNone(gate.criteria["hit_rate"]["shadow"])
        self.assertFalse(gate.criteria["hit_rate"]["passed"])
        self.assertFalse(gate.eligible)

    def test_challenger_rows_without_production_rows_after_the_change_do_not_win(self):
        """challenger だけ変わり目より後の行があっても、production が前のモデルの行なら勝たない。"""
        early = _rows("production", [False] * _N_DAYS)
        _insert(early[early["predicted_at"] <= self.change.strftime("%Y%m%d_%H%M%S")])
        _change_production(self.models, self.change)

        result = self._shadow()

        self.assertFalse(result["challenger_wins"])
        self.assertEqual(result["n_production"], 0)
        self.assertEqual(result["n_challenger"], _N_AFTER_CHANGE * _N_SYMBOLS)

    def test_change_before_challenger_training_keeps_the_challenger_window(self):
        _insert(_rows("production", self.prod_hits))
        _change_production(self.models, _after_day(_N_DAYS - _N_CHALLENGER_DAYS - 3))

        result = self._shadow()

        self.assertEqual(result["window_days"], _N_CHALLENGER_DAYS)
        self.assertEqual(result["n_challenger"], _N_CHALLENGER_DAYS * _N_SYMBOLS)
        self.assertEqual(result["n_production"], _N_CHALLENGER_DAYS * _N_SYMBOLS)

    def test_without_a_change_the_window_stays_the_challenger_window(self):
        _insert(_rows("production", self.prod_hits))

        result = self._shadow()

        self.assertEqual(result["window_days"], _N_CHALLENGER_DAYS)
        self.assertAlmostEqual(result["challenger_hit_rate"], 6 / _N_CHALLENGER_DAYS)
        self.assertAlmostEqual(result["production_hit_rate"], 2 / _N_CHALLENGER_DAYS)

    def test_accuracy_report_filter_ignores_the_production_change(self):
        """採点の JSON の challenger の欄はモデルが同じなので、今の窓のままにする。"""
        from src.prediction.challenger_window import filter_current_challenger

        _change_production(self.models, self.change)
        rows = _rows("challenger", self.chal_hits)

        kept, window = filter_current_challenger(rows)

        self.assertEqual(len(kept), _N_CHALLENGER_DAYS * _N_SYMBOLS)
        self.assertEqual(len(window.days), _N_CHALLENGER_DAYS)

    def test_short_window_is_kept_in_the_weekly_record(self):
        from src.prediction.shadow_record import apply_shadow_result, new_weekly_shadow_record

        _insert(_rows("production", self.prod_hits))
        _change_production(self.models, self.change)
        record = new_weekly_shadow_record(days=_N_DAYS)

        apply_shadow_result(record, self._shadow())

        self.assertEqual(record["window_days"], _N_AFTER_CHANGE)
        self.assertEqual(record["production_changed_at"], self.change.isoformat())


class TestSameTradingDayAroundTheChange(_ModelsDirTestCase):
    def test_production_row_before_the_change_is_dropped_on_the_same_trading_day(self):
        """引けの後の回と翌朝の回は同じ取引日になる。変わり目の前の production の回を残さない。"""
        from src.prediction.challenger_window import filter_comparable_challenger, filter_to_days

        day = _DAYS[-1]
        evening = day.strftime("%Y%m%d_100000")
        morning = day.strftime("%Y%m%d_224500")
        when = datetime(day.year, day.month, day.day, 13, 0, tzinfo=timezone.utc)
        _write_challenger(self.models, when - timedelta(days=3))
        _write_production(self.models)
        _change_production(self.models, when)
        prod = pd.concat(
            [_rows("production", [False], "100000"), _rows("production", [True], "224500")],
            ignore_index=True,
        )

        _, window = filter_comparable_challenger(_rows("challenger", [True], "224500"))
        kept = filter_to_days(prod, window)

        self.assertEqual(set(kept["predicted_at"]), {morning})
        self.assertNotIn(evening, set(kept["predicted_at"]))
        self.assertEqual(window.n_production, _N_SYMBOLS)


if __name__ == "__main__":
    unittest.main()
