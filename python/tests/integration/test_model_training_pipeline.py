"""model_training_pipeline モジュールのユニットテスト"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.prediction.training_pipeline import train_models_for_symbol, train_models_for_symbol_task
from src.prediction.types import FeatureLoadResult


class TestTrainModelsForSymbol(unittest.TestCase):
    """train_models_for_symbol 関数のテスト"""

    def setUp(self):
        """テスト用一時DBを設定"""
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_training.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal_path = self.tmp_db + ".wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_skip_when_no_data(self):
        """DBにデータがない場合にskipステータスが返ることを確認"""
        result = train_models_for_symbol("us", "NODATA")
        self.assertEqual(result["status"], "skip")
        self.assertEqual(result["market"], "us")
        self.assertEqual(result["symbol"], "NODATA")

    def test_skip_with_empty_data(self):
        """DBに空データがある場合にskipステータスが返ることを確認"""
        with patch(
            "src.prediction.training_pipeline._features.load_stock_features",
            return_value=pd.DataFrame(),
        ):
            result = train_models_for_symbol("us", "EMPTY")
        self.assertEqual(result["status"], "skip")

    @patch("src.prediction.training_pipeline._training.ModelManager")
    @patch("src.prediction.training_pipeline._features.load_stock_features")
    def test_success_trains_both_models(self, mock_load, mock_mm_cls):
        """正常なデータでXGBoostとLightGBMの両方が学習されることを確認"""
        # テスト用DataFrameを返す
        df = pd.DataFrame(
            {
                "feat1": np.random.rand(50),
                "feat2": np.random.rand(50),
                "y": np.random.rand(50),
            }
        )
        mock_load.return_value = df

        mock_mm = MagicMock()
        mock_mm_cls.return_value = mock_mm

        result = train_models_for_symbol("us", "AAPL")

        self.assertEqual(result["status"], "success")
        # create_modelが2回呼ばれる（XGBoost + LightGBM）
        self.assertEqual(mock_mm.create_model.call_count, 2)
        # train_modelが2回呼ばれる
        self.assertEqual(mock_mm.train_model.call_count, 2)

    @patch("src.prediction.training_pipeline._training.ModelManager")
    @patch("src.prediction.training_pipeline._features.load_stock_features")
    def test_error_returns_error_status(self, mock_load, mock_mm_cls):
        """学習中に例外が発生した場合にerrorステータスが返ることを確認"""
        df = pd.DataFrame(
            {
                "feat1": np.random.rand(20),
                "y": np.random.rand(20),
            }
        )
        mock_load.return_value = df
        mock_mm = MagicMock()
        mock_mm.create_model.side_effect = RuntimeError("学習エラー")
        mock_mm_cls.return_value = mock_mm

        result = train_models_for_symbol("us", "FAIL")
        self.assertEqual(result["status"], "error")
        self.assertIn("学習エラー", result["error"])

    @patch("src.prediction.training_pipeline._training.ModelManager")
    @patch("src.prediction.training_pipeline._features.load_stock_features")
    def test_excludes_string_columns(self, mock_load, mock_mm_cls):
        """market, symbol, y列が特徴量から除外されることを確認"""
        df = pd.DataFrame(
            {
                "feat1": np.random.rand(20),
                "market": ["us"] * 20,
                "symbol": ["AAPL"] * 20,
                "y": np.random.rand(20),
            }
        )
        mock_load.return_value = df
        mock_mm = MagicMock()
        mock_mm_cls.return_value = mock_mm

        result = train_models_for_symbol("us", "AAPL")
        self.assertEqual(result["status"], "success")
        # train_modelに渡されたXにmarket/symbol/y列が含まれていないことを確認
        call_args = mock_mm.train_model.call_args_list[0]
        X_passed = call_args[0][1]  # 第2引数がX
        self.assertNotIn("market", X_passed.columns)
        self.assertNotIn("symbol", X_passed.columns)
        self.assertNotIn("y", X_passed.columns)


class TestTrainModelsForSymbolTask(unittest.TestCase):
    """train_models_for_symbol_task ラッパー関数のテスト"""

    def test_unpacks_dict_correctly(self):
        """dict引数が正しく展開されて呼び出されることを確認"""
        with patch("src.prediction.training_pipeline._training.train_models_for_symbol") as mock_fn:
            mock_fn.return_value = {"status": "success"}
            result = train_models_for_symbol_task({"market": "jp", "symbol": "7203"})
            mock_fn.assert_called_once_with(
                "jp", "7203", 1, shadow_mode=False, use_transformer=False
            )
            self.assertEqual(result["status"], "success")


class TestRunBatchTraining(unittest.TestCase):
    """run_batch_training 関数のテスト"""

    def setUp(self):
        """テスト用一時DBを設定"""
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_model_batch.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal_path = self.tmp_db + ".wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    @patch("src.prediction.training_pipeline._training.load_features_for_training")
    def test_batch_success(self, mock_load):
        """バッチ処理が正常に完了することを確認"""
        from src.domain.types import SymbolTask
        from src.prediction.training_pipeline import run_batch_training

        X = pd.DataFrame(np.random.rand(30, 5), columns=[f"feat{i}" for i in range(5)])
        y = pd.Series(np.random.rand(30))
        mock_load.return_value = FeatureLoadResult(
            market="us", symbol="TEST1", status="success", X=X, y=y
        )

        tasks = [SymbolTask(market="us", symbol="TEST1")]
        result = run_batch_training(tasks=tasks)

        self.assertGreaterEqual(len(result.succeeded) + len(result.failed), 1)

    def test_batch_no_tasks(self):
        """対象銘柄がない場合、空の BatchResult が返ること"""
        from src.prediction.training_pipeline import run_batch_training

        result = run_batch_training(tasks=[])

        self.assertEqual(len(result.succeeded), 0)
        self.assertEqual(len(result.failed), 0)

    @patch("src.prediction.training_pipeline._training.ModelManager")
    @patch("src.prediction.training_pipeline._training.load_features_for_training")
    def test_batch_with_training_errors(self, mock_load, mock_mm_cls):
        """フェーズ2で学習エラーが発生しても例外が外に出ないこと"""
        from src.domain.types import SymbolTask
        from src.prediction.training_pipeline import run_batch_training

        X = pd.DataFrame(np.random.rand(30, 5), columns=[f"feat{i}" for i in range(5)])
        y = pd.Series(np.random.rand(30))
        mock_load.return_value = FeatureLoadResult(
            market="us", symbol="TEST1", status="success", X=X, y=y
        )
        mock_mm = MagicMock()
        mock_mm.train_model.side_effect = RuntimeError("学習失敗")
        mock_mm_cls.return_value = mock_mm

        tasks = [SymbolTask(market="us", symbol="TEST1")]
        result = run_batch_training(tasks=tasks)  # 例外が外に出ないこと

        self.assertEqual(len(result.failed), 1)

    @patch("src.prediction.training_pipeline._training.load_features_for_training")
    def test_batch_with_load_errors(self, mock_load):
        """フェーズ1でデータ読み込みエラーが発生する場合を確認"""
        from src.domain.types import SymbolTask
        from src.prediction.training_pipeline import run_batch_training

        mock_load.return_value = FeatureLoadResult(
            market="us", symbol="TEST1", status="error", error="DB エラー"
        )

        tasks = [SymbolTask(market="us", symbol="TEST1")]
        result = run_batch_training(tasks=tasks)

        self.assertEqual(len(result.failed), 1)
        self.assertEqual(len(result.succeeded), 0)


if __name__ == "__main__":
    unittest.main()
