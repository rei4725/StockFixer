"""save_optimal_params_json と print_optimization_results が最良の行を選ぶこと（Issue 1018）。

max_drawdown は負の小数（metrics.stats.max_drawdown）で、0 に近いほど良い。
cost_impact_* は gross - net（コストで失った分）で、小さいほど良い。
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

from src.backtest.optimizer import persistence


def _result_df() -> pd.DataFrame:
    # 行の並びは、どの指標でも最良が先頭・最後のどちらでもない位置に来るようにする
    return pd.DataFrame(
        {
            "threshold": [0.01, 0.02, 0.03],
            "sharpe_ratio": [0.8, 1.5, 1.0],
            "max_drawdown": [-0.30, -0.20, -0.05],
            "cost_impact_return": [0.02, 0.05, 0.01],
            "cost_impact_cash": [300.0, 100.0, 500.0],
            "num_trades": [40, 50, 60],
            "error": [None, None, None],
        }
    )


class TestSaveOptimalParamsBestRow(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        fake_file = os.path.join(root, "src", "backtest", "optimizer", "persistence.py")
        file_patch = patch.object(persistence, "__file__", fake_file)
        file_patch.start()
        self.addCleanup(file_patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def _saved_threshold(self, sort_by: str) -> float:
        with redirect_stdout(io.StringIO()):
            path = persistence.save_optimal_params_json(_result_df(), "jp", "1301", sort_by=sort_by)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["jp_1301"]["threshold"]

    def test_sharpe_ratio_saves_largest(self):
        self.assertEqual(self._saved_threshold("sharpe_ratio"), 0.02)

    def test_max_drawdown_saves_closest_to_zero(self):
        self.assertEqual(self._saved_threshold("max_drawdown"), 0.03)

    def test_cost_impact_return_saves_smallest(self):
        self.assertEqual(self._saved_threshold("cost_impact_return"), 0.03)

    def test_cost_impact_cash_saves_smallest(self):
        self.assertEqual(self._saved_threshold("cost_impact_cash"), 0.02)

    def test_error_rows_are_skipped(self):
        df = _result_df()
        df.loc[1, "error"] = "failed"
        with redirect_stdout(io.StringIO()):
            path = persistence.save_optimal_params_json(df, "jp", "1301", sort_by="sharpe_ratio")
        with open(path, "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["jp_1301"]["threshold"], 0.03)


class TestPrintBestMatchesSaved(unittest.TestCase):
    def _printed_best_threshold(self, sort_by: str) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            persistence.print_optimization_results(_result_df(), sort_by)
        best_part = buf.getvalue().split("ベスト")[1]
        line = next(ln for ln in best_part.splitlines() if "閾値" in ln)
        return line.split(":", 1)[1].strip()

    def test_print_best_matches_saved_row(self):
        expected = {
            "sharpe_ratio": "0.02",
            "max_drawdown": "0.03",
            "cost_impact_return": "0.03",
            "cost_impact_cash": "0.02",
        }
        for sort_by, threshold in expected.items():
            with self.subTest(sort_by=sort_by):
                self.assertEqual(self._printed_best_threshold(sort_by), threshold)


if __name__ == "__main__":
    unittest.main()
