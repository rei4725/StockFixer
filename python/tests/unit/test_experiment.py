"""ユニットテスト: experiment.py (R-211 実験トラッキング基盤)"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.utils.db.experiment import (
    _feature_hash,
    generate_run_id,
    load_best_run,
    load_experiment_runs,
    save_experiment_run,
)

# ---------------------------------------------------------------------------
# generate_run_id / _feature_hash
# ---------------------------------------------------------------------------


class TestGenerateRunId(unittest.TestCase):
    def test_returns_unique_ids(self):
        ids = {generate_run_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_returns_string(self):
        self.assertIsInstance(generate_run_id(), str)


class TestFeatureHash(unittest.TestCase):
    def test_same_features_same_hash(self):
        feats = ["close", "volume", "sma5"]
        self.assertEqual(_feature_hash(feats), _feature_hash(feats))

    def test_order_independent(self):
        self.assertEqual(_feature_hash(["a", "b", "c"]), _feature_hash(["c", "a", "b"]))

    def test_different_features_different_hash(self):
        self.assertNotEqual(_feature_hash(["a", "b"]), _feature_hash(["a", "c"]))

    def test_returns_16_char_hex(self):
        h = _feature_hash(["feat1", "feat2"])
        self.assertEqual(len(h), 16)
        int(h, 16)  # 16進数として解釈できること


# ---------------------------------------------------------------------------
# save_experiment_run
# ---------------------------------------------------------------------------


class TestSaveExperimentRun(unittest.TestCase):
    def _make_mock_con(self):
        con = MagicMock()
        return con

    def test_calls_insert_with_correct_run_id(self):
        mock_con = self._make_mock_con()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            save_experiment_run(
                run_id="test-run-001",
                market="jp",
                symbol="7203",
                model_name="StockXGBoostModel",
                trained_at="20260314_120000",
                horizon=1,
                rmse=0.012,
                directional_accuracy=0.58,
                n_samples=500,
                feature_names=["close", "volume"],
            )

        mock_con.execute.assert_called_once()
        call_args = mock_con.execute.call_args
        params = call_args[0][1]
        self.assertEqual(params[0], "test-run-001")
        self.assertEqual(params[1], "jp")
        self.assertEqual(params[2], "7203")

    def test_feature_hash_computed(self):
        mock_con = self._make_mock_con()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            save_experiment_run(
                run_id="run-hash-test",
                market="us",
                symbol="AAPL",
                model_name="StockLightGBMModel",
                trained_at="20260314_130000",
                feature_names=["feature_a", "feature_b"],
            )

        params = mock_con.execute.call_args[0][1]
        # n_features = 2, feature_hash = 16文字
        self.assertEqual(params[9], 2)
        self.assertIsNotNone(params[10])
        self.assertEqual(len(params[10]), 16)

    def test_none_feature_names_saves_null(self):
        mock_con = self._make_mock_con()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            save_experiment_run(
                run_id="run-no-feat",
                market="jp",
                symbol="6758",
                model_name="StockXGBoostModel",
                trained_at="20260314_140000",
                feature_names=None,
            )

        params = mock_con.execute.call_args[0][1]
        self.assertIsNone(params[9])  # n_features
        self.assertIsNone(params[10])  # feature_hash

    def test_params_serialized_as_json(self):
        mock_con = self._make_mock_con()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)

        import json

        test_params = {"learning_rate": 0.05, "n_estimators": 200}
        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            save_experiment_run(
                run_id="run-params",
                market="jp",
                symbol="7203",
                model_name="StockXGBoostModel",
                trained_at="20260314_150000",
                params=test_params,
            )

        params = mock_con.execute.call_args[0][1]
        stored = json.loads(params[11])  # params_json
        self.assertEqual(stored["learning_rate"], 0.05)
        self.assertEqual(stored["n_estimators"], 200)


# ---------------------------------------------------------------------------
# load_experiment_runs
# ---------------------------------------------------------------------------


class TestLoadExperimentRuns(unittest.TestCase):
    def _make_mock_with_df(self, df: pd.DataFrame):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchdf.return_value = df
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)
        return mock_cm

    def test_returns_dataframe(self):
        expected = pd.DataFrame(
            [
                {
                    "run_id": "r1",
                    "market": "jp",
                    "symbol": "7203",
                    "model_name": "StockXGBoostModel",
                }
            ]
        )
        mock_cm = self._make_mock_with_df(expected)
        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            result = load_experiment_runs(market="jp", symbol="7203")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    def test_returns_empty_df_on_error(self):
        mock_con = MagicMock()
        mock_con.execute.side_effect = Exception("table does not exist")
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            result = load_experiment_runs()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_no_filter_uses_no_where_conditions(self):
        mock_cm = self._make_mock_with_df(pd.DataFrame())
        mock_con = mock_cm.__enter__.return_value
        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            load_experiment_runs()
        query = mock_con.execute.call_args[0][0]
        self.assertIn("WHERE 1=1", query)
        self.assertNotIn("AND market", query)

    def test_filter_by_model_name(self):
        mock_cm = self._make_mock_with_df(pd.DataFrame())
        mock_con = mock_cm.__enter__.return_value
        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            load_experiment_runs(model_name="StockXGBoostModel")
        query = mock_con.execute.call_args[0][0]
        self.assertIn("AND model_name = ?", query)


# ---------------------------------------------------------------------------
# load_best_run
# ---------------------------------------------------------------------------


class TestLoadBestRun(unittest.TestCase):
    def _make_mock_with_row(self, row_dict: dict | None):
        df = pd.DataFrame([row_dict]) if row_dict else pd.DataFrame()
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchdf.return_value = df
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)
        return mock_cm

    def test_returns_dict_when_found(self):
        mock_cm = self._make_mock_with_row(
            {
                "run_id": "r1",
                "market": "jp",
                "symbol": "7203",
                "directional_accuracy": 0.62,
            }
        )
        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            result = load_best_run("jp", "7203", "StockXGBoostModel")
        self.assertIsNotNone(result)
        self.assertEqual(result["run_id"], "r1")

    def test_returns_none_when_empty(self):
        mock_cm = self._make_mock_with_row(None)
        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            result = load_best_run("jp", "7203", "StockXGBoostModel")
        self.assertIsNone(result)

    def test_rmse_uses_asc_order(self):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            load_best_run("jp", "7203", "StockXGBoostModel", metric="rmse")
        query = mock_con.execute.call_args[0][0]
        self.assertIn("ASC", query)

    def test_directional_accuracy_uses_desc_order(self):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchdf.return_value = pd.DataFrame()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            load_best_run("jp", "7203", "StockXGBoostModel", metric="directional_accuracy")
        query = mock_con.execute.call_args[0][0]
        self.assertIn("DESC", query)

    def test_returns_none_on_error(self):
        mock_con = MagicMock()
        mock_con.execute.side_effect = Exception("db error")
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_con)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("src.utils.db.experiment._db_connection", return_value=mock_cm):
            result = load_best_run("jp", "7203", "StockXGBoostModel")
        self.assertIsNone(result)

    def test_raises_on_invalid_metric(self):
        with self.assertRaises(ValueError):
            load_best_run(
                "jp", "7203", "StockXGBoostModel", metric="DROP TABLE experiment_runs; --"
            )


if __name__ == "__main__":
    unittest.main()
