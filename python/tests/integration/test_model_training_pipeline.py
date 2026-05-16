"""model_training_pipeline モジュールのユニットテスト"""

import os
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
            os.rmdir(self.tmp_dir)

    def test_skip_when_no_data(self):
        """DBにデータがない場合にskipステータスが返ることを確認"""
        result = train_models_for_symbol("us", "NODATA")
        self.assertEqual(result["status"], "skip")
        self.assertEqual(result["market"], "us")
        self.assertEqual(result["symbol"], "NODATA")

    def test_skip_with_empty_data(self):
        """DBに空データがある場合にskipステータスが返ることを確認"""
        with patch(
            "src.prediction.training_pipeline.load_stock_features", return_value=pd.DataFrame()
        ):
            result = train_models_for_symbol("us", "EMPTY")
        self.assertEqual(result["status"], "skip")

    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.prediction.training_pipeline.load_stock_features")
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

    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.prediction.training_pipeline.load_stock_features")
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

    @patch("src.prediction.training_pipeline.ModelManager")
    @patch("src.prediction.training_pipeline.load_stock_features")
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
        with patch("src.prediction.training_pipeline.train_models_for_symbol") as mock_fn:
            mock_fn.return_value = {"status": "success"}
            result = train_models_for_symbol_task({"market": "jp", "symbol": "7203"})
            mock_fn.assert_called_once_with("jp", "7203", 1)
            self.assertEqual(result["status"], "success")


class TestRunModelBatch(unittest.TestCase):
    """run_model_batch 関数のテスト"""

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
            os.rmdir(self.tmp_dir)

    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.print_summary")
    def test_batch_success(
        self,
        mock_print_summary,
        mock_run_parallel,
        mock_load_symbols,
    ):
        """バッチ処理が正常に完了することを確認"""
        from src.prediction.training_pipeline import run_model_batch

        # テスト用シンボル
        mock_load_symbols.return_value = [
            {"market": "us", "symbol": "TEST1"},
            {"market": "us", "symbol": "TEST2"},
        ]

        # フェーズ1の結果（データ読み込みのみ）
        X = pd.DataFrame(np.random.rand(30, 5), columns=[f"feat{i}" for i in range(5)])
        y = pd.Series(np.random.rand(30))

        phase1_results = [
            FeatureLoadResult(market="us", symbol="TEST1", status="success", X=X, y=y),
            FeatureLoadResult(market="us", symbol="TEST2", status="success", X=X, y=y),
        ]
        mock_run_parallel.return_value = phase1_results

        # 実行
        run_model_batch()

        # run_parallelが呼ばれたことを確認
        mock_run_parallel.assert_called_once()
        # print_summaryが呼ばれたことを確認
        mock_print_summary.assert_called_once()

    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.watchlist.batch_runner.print_summary")
    def test_batch_no_symbols(
        self,
        mock_print_summary,
        mock_load_symbols,
    ):
        """対象銘柄がない場合を確認"""
        from src.prediction.training_pipeline import run_model_batch

        mock_load_symbols.return_value = []

        # 実行（例外が発生しないことを確認）
        run_model_batch()

        # print_summaryは呼ばれないはず
        mock_print_summary.assert_not_called()

    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.print_summary")
    @patch("src.prediction.training_pipeline.ModelManager")
    def test_batch_with_training_errors(
        self,
        mock_mm_cls,
        mock_print_summary,
        mock_run_parallel,
        mock_load_symbols,
    ):
        """フェーズ2で学習エラーが発生する場合を確認"""
        from src.prediction.training_pipeline import run_model_batch

        mock_load_symbols.return_value = [
            {"market": "us", "symbol": "TEST1"},
        ]

        # フェーズ1の結果（成功）
        X = pd.DataFrame(np.random.rand(30, 5), columns=[f"feat{i}" for i in range(5)])
        y = pd.Series(np.random.rand(30))
        phase1_results = [
            FeatureLoadResult(market="us", symbol="TEST1", status="success", X=X, y=y),
        ]
        mock_run_parallel.return_value = phase1_results

        # フェーズ2で学習エラーをシミュレート
        mock_mm = MagicMock()
        mock_mm.train_model.side_effect = RuntimeError("学習失敗")
        mock_mm_cls.return_value = mock_mm

        # 実行（例外が発生しないことを確認）
        run_model_batch()

        # print_summaryが呼ばれたことを確認（エラーサマリーを含む）
        mock_print_summary.assert_called_once()

    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.print_summary")
    def test_batch_with_load_errors(
        self,
        mock_print_summary,
        mock_run_parallel,
        mock_load_symbols,
    ):
        """フェーズ1でデータ読み込みエラーが発生する場合を確認"""
        from src.prediction.training_pipeline import run_model_batch

        mock_load_symbols.return_value = [
            {"market": "us", "symbol": "TEST1"},
            {"market": "us", "symbol": "TEST2"},
        ]

        # フェーズ1の結果（1つ成功、1つエラー）
        X = pd.DataFrame(np.random.rand(30, 5), columns=[f"feat{i}" for i in range(5)])
        y = pd.Series(np.random.rand(30))

        phase1_results = [
            FeatureLoadResult(market="us", symbol="TEST1", status="success", X=X, y=y),
            FeatureLoadResult(market="us", symbol="TEST2", status="error", error="読み込み失敗"),
        ]
        mock_run_parallel.return_value = phase1_results

        # 実行
        run_model_batch()

        # print_summaryが呼ばれたことを確認
        mock_print_summary.assert_called_once()


if __name__ == "__main__":
    unittest.main()
