import os
import tempfile
import unittest

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.utils.db.experiment import load_experiment_runs, save_experiment_run


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
    def test_save_and_load_roundtrip(self):
        save_experiment_run(
            run_name="baseline",
            hyperparameters={"max_depth": 6},
            metrics={"rmse": 0.12},
            features=["close", "volume"],
        )
        rows = load_experiment_runs()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_name"], "baseline")

    def test_hyperparameters_roundtrip(self):
        save_experiment_run(
            run_name="test_hp",
            hyperparameters={"lr": 0.01, "n_estimators": 100},
            metrics={"auc": 0.95},
            features=["open", "high", "low"],
        )
        rows = load_experiment_runs()
        self.assertEqual(rows[0]["hyperparameters"]["lr"], 0.01)
        self.assertEqual(rows[0]["hyperparameters"]["n_estimators"], 100)

    def test_metrics_roundtrip(self):
        save_experiment_run(
            run_name="test_metrics",
            hyperparameters={},
            metrics={"rmse": 0.05, "mae": 0.03},
            features=[],
        )
        rows = load_experiment_runs()
        self.assertAlmostEqual(rows[0]["metrics"]["rmse"], 0.05)

    def test_features_roundtrip(self):
        save_experiment_run(
            run_name="test_features",
            hyperparameters={},
            metrics={},
            features=["f1", "f2", "f3"],
        )
        result = load_experiment_runs()
        self.assertEqual(result[0]["features"], ["f1", "f2", "f3"])

    def test_load_limit(self):
        for i in range(5):
            save_experiment_run(run_name=f"run_{i}", hyperparameters={}, metrics={}, features=[])
        rows = load_experiment_runs(limit=3)
        self.assertEqual(len(rows), 3)
