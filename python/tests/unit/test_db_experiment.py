import os
import tempfile
import unittest

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.utils.db.experiment import (
    generate_run_id,
    load_best_run,
    load_experiment_runs,
    save_experiment_run,
)


class _TmpDbTestCase(unittest.TestCase):
    def setUp(self):
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path


class TestExperimentDb(_TmpDbTestCase):
    def _make_run(
        self,
        run_id=None,
        market="us",
        symbol="AAPL",
        model_name="xgboost",
        trained_at="20260501_120000",
        **kwargs,
    ):
        if run_id is None:
            run_id = generate_run_id()
        save_experiment_run(
            run_id=run_id,
            market=market,
            symbol=symbol,
            model_name=model_name,
            trained_at=trained_at,
            **kwargs,
        )
        return run_id

    def test_save_and_load_roundtrip(self):
        self._make_run(run_id="test-run-1")
        df = load_experiment_runs()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["run_id"], "test-run-1")

    def test_rmse_roundtrip(self):
        self._make_run(run_id="test-rmse", rmse=0.12)
        df = load_experiment_runs()
        self.assertAlmostEqual(float(df.iloc[0]["rmse"]), 0.12)

    def test_directional_accuracy_roundtrip(self):
        self._make_run(run_id="test-da", directional_accuracy=0.65)
        df = load_experiment_runs()
        self.assertAlmostEqual(float(df.iloc[0]["directional_accuracy"]), 0.65)

    def test_feature_names_roundtrip(self):
        self._make_run(run_id="test-feat", feature_names=["f1", "f2", "f3"])
        df = load_experiment_runs()
        self.assertEqual(int(df.iloc[0]["n_features"]), 3)

    def test_load_limit(self):
        for i in range(5):
            self._make_run(run_id=f"run-{i}", symbol=f"SYM{i}")
        df = load_experiment_runs(limit=3)
        self.assertEqual(len(df), 3)

    def test_market_filter(self):
        self._make_run(run_id="us-run", market="us", symbol="AAPL")
        self._make_run(run_id="jp-run", market="jp", symbol="7203")
        df = load_experiment_runs(market="jp")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["market"], "jp")

    def test_symbol_filter(self):
        self._make_run(run_id="run-aapl", market="us", symbol="AAPL")
        self._make_run(run_id="run-goog", market="us", symbol="GOOG")
        df = load_experiment_runs(symbol="GOOG")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "GOOG")

    def test_load_best_run_by_directional_accuracy(self):
        self._make_run(run_id="run-a", directional_accuracy=0.55)
        self._make_run(run_id="run-b", directional_accuracy=0.72)
        best = load_best_run("us", "AAPL", "xgboost", metric="directional_accuracy")
        self.assertIsNotNone(best)
        self.assertEqual(best["run_id"], "run-b")

    def test_load_best_run_by_rmse(self):
        self._make_run(run_id="run-c", rmse=0.20)
        self._make_run(run_id="run-d", rmse=0.05)
        best = load_best_run("us", "AAPL", "xgboost", metric="rmse")
        self.assertIsNotNone(best)
        self.assertEqual(best["run_id"], "run-d")

    def test_load_best_run_no_data(self):
        result = load_best_run("us", "AAPL", "xgboost")
        self.assertIsNone(result)
