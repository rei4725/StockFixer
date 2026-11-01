"""手で流す銘柄別のモデルの回が、日次の production の採点に混ざらない（#1457）。

prediction_results は (market, symbol, model_version) ごとに Delete-Insert で1行だけを
持つ。銘柄別のモデルの回を production で保存すると統合モデルの行が上書きされ、翌日の
[2/6] の採点（model_version_filter="production"）が銘柄別のモデルの予想を採点する。
"""

import os
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import run_predict

from src.prediction import prediction_pipeline as pipeline
from src.prediction.types import PredictionResult

_UNIFIED_PRICE = 110.0
_INDIVIDUAL_PRICE = 90.0
_SYMBOLS = ["7203", "6758", "9984"]


def _result(symbol: str, pred_price: float) -> PredictionResult:
    return PredictionResult(
        market="jp",
        symbol=symbol,
        current_price=100.0,
        avg_pred_price=pred_price,
        diff_ratio=(pred_price - 100.0) / 100.0,
        model_count=2,
    )


class _FakePredictionTable:
    """prediction_results の Delete-Insert と model_version の絞り込みだけをまねる。"""

    def __init__(self):
        self.rows: list[dict] = []

    def save(self, predicted_at: str, results: list[PredictionResult]) -> None:
        for r in results:
            version = r.model_version or "production"
            self.rows = [
                row
                for row in self.rows
                if (row["market"], row["symbol"], row["model_version"])
                != (r.market, r.symbol, version)
            ]
            self.rows.append(
                {
                    "market": r.market,
                    "symbol": r.symbol,
                    "predicted_at": predicted_at,
                    "model_version": version,
                    "current_price": r.current_price,
                    "avg_pred_price": r.avg_pred_price,
                }
            )

    def load(self, limit=None, model_version=None, raise_on_error=False, **_kwargs):
        rows = [
            r for r in self.rows if model_version is None or r["model_version"] == model_version
        ]
        return pd.DataFrame(rows)


def _ohlcv(market: str, symbol: str) -> pd.DataFrame:
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    index = pd.DatetimeIndex([today + timedelta(days=d) for d in range(-5, 6)])
    return pd.DataFrame({"Close": [100.0] * len(index)}, index=index)


class TestIndividualRunNotScored(unittest.TestCase):
    def setUp(self):
        self.table = _FakePredictionTable()
        self.scored: list[dict] = []
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.watchlist = os.path.join(self._tmp.name, "watchlist.csv")
        with open(self.watchlist, "w", encoding="utf-8") as f:
            f.write("".join(f"jp,{s},name\n" for s in _SYMBOLS))

        stack = ExitStack()
        self.addCleanup(stack.close)

        def p(*args, **kwargs):
            return stack.enter_context(patch(*args, **kwargs))

        p("src.prediction.prediction_pipeline.save_prediction_results", self.table.save)
        p("src.prediction.db.load_prediction_results", self.table.load)
        p("src.prediction.db.save_prediction_accuracy", side_effect=self._save_accuracy)
        p("src.prediction.db.load_drift_summary", return_value=pd.DataFrame())
        p("src.utils.db.load_raw_ohlcv", side_effect=_ohlcv)
        p(
            "src.prediction.prediction_pipeline.predict_single_stock",
            side_effect=lambda market, symbol, **_k: _result(symbol, _INDIVIDUAL_PRICE),
        )
        p("src.utils.data_path_utils.get_monitor_list_path", return_value=self.watchlist)
        p("src.orchestration.jobs.prediction_saved_marker.clear_prediction_saved_marker")
        p(
            "run_predict.predict_all_individual",
            return_value=[_result(s, _INDIVIDUAL_PRICE) for s in _SYMBOLS],
        )

        # 日次の [3/6] と同じく、統合モデルの回を production で保存する
        pipeline.output_top_worst_results(
            [_result(s, _UNIFIED_PRICE) for s in _SYMBOLS], mode="unified"
        )

    def _save_accuracy(self, rows: list[dict]) -> int:
        self.scored.extend(rows)
        return len(rows)

    def _daily_scoring(self) -> None:
        pipeline.run_accuracy_check(
            horizon=1, model_name="production", model_version_filter="production"
        )

    def _assert_only_unified_scored(self) -> None:
        self.assertEqual(len(self.scored), len(_SYMBOLS))
        self.assertEqual({r["predicted_price"] for r in self.scored}, {_UNIFIED_PRICE})

    def test_individual_top10_not_scored_as_production(self):
        run_predict.run_top10(use_individual=True)
        self._daily_scoring()
        self._assert_only_unified_scored()

    def test_single_not_scored_as_production(self):
        pipeline.run_predict_single("jp", "7203")
        self._daily_scoring()
        self._assert_only_unified_scored()

    def test_watchlist_not_scored_as_production(self):
        pipeline.run_predict_watchlist()
        self._daily_scoring()
        self._assert_only_unified_scored()

    def test_individual_runs_keep_own_label(self):
        """銘柄別のモデルの回は消さずに別のラベルで残す"""
        run_predict.run_top10(use_individual=True)
        individual = self.table.load(model_version=pipeline.INDIVIDUAL_MODEL_VERSION)
        self.assertEqual(set(individual["symbol"]), set(_SYMBOLS))
        self.assertEqual(set(individual["avg_pred_price"]), {_INDIVIDUAL_PRICE})


if __name__ == "__main__":
    unittest.main()
