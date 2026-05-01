"""
DuckDB アクセスモジュールのユニットテスト
"""

import os
import tempfile
import unittest

import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.prediction.types import PredictionResult


class TestDB(unittest.TestCase):
    """db.py の CRUD 操作テスト"""

    def setUp(self):
        """各テスト前にインメモリDBで初期化"""
        # 既存接続をクリア
        db_module.close_connection()
        # テスト用の一時DBファイルを作成
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test.duckdb")
        # get_db_path をモンキーパッチ
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        # 接続を強制的にリセット
        db_module._tables_initialized = False

    def tearDown(self):
        """各テスト後にDBをクリーンアップ"""
        db_module.close_connection()
        # モンキーパッチを元に戻す
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        # 一時ファイル削除
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal_path = self.tmp_db + ".wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def test_init_tables(self):
        """テーブルが正しく作成されることを確認"""
        con = db_module.get_connection()
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {row[0] for row in tables}
        self.assertIn("stock_features", table_names)
        self.assertIn("prediction_results", table_names)
        self.assertIn("index_membership_history", table_names)

    def test_upsert_and_load_stock_features(self):
        """stock_features の upsert と load が正しく動作することを確認"""
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [95.0, 96.0, 97.0],
                "Close": [103.0, 104.0, 105.0],
                "Volume": [1000, 2000, 3000],
                "y": [104.0, 105.0, 106.0],
            }
        )

        db_module.upsert_stock_features("us", "AAPL", df)

        loaded = db_module.load_stock_features("us", "AAPL")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded), 3)
        self.assertIn("Open", loaded.columns)
        self.assertIn("Close", loaded.columns)
        self.assertIn("y", loaded.columns)
        # market, symbol, row_num は除外されている
        self.assertNotIn("market", loaded.columns)
        self.assertNotIn("symbol", loaded.columns)
        self.assertNotIn("row_num", loaded.columns)

    def test_upsert_replaces_existing_data(self):
        """同じ market/symbol で upsert すると既存データが置き換えられることを確認"""
        df1 = pd.DataFrame({"Close": [100.0, 101.0], "y": [102.0, 103.0]})
        db_module.upsert_stock_features("us", "AAPL", df1)

        df2 = pd.DataFrame({"Close": [200.0], "y": [201.0]})
        db_module.upsert_stock_features("us", "AAPL", df2)

        loaded = db_module.load_stock_features("us", "AAPL")
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(loaded["Close"].iloc[0], 200.0)

    def test_load_stock_features_nonexistent(self):
        """存在しない銘柄を load すると None が返ることを確認"""
        loaded = db_module.load_stock_features("us", "NONEXIST")
        self.assertIsNone(loaded)

    def test_load_all_stock_features(self):
        """全銘柄の特徴量が取得できることを確認"""
        df1 = pd.DataFrame({"Close": [100.0], "y": [101.0]})
        df2 = pd.DataFrame({"Close": [200.0], "y": [201.0]})
        db_module.upsert_stock_features("us", "AAPL", df1)
        db_module.upsert_stock_features("jp", "7203", df2)

        all_df = db_module.load_all_stock_features()
        self.assertEqual(len(all_df), 2)
        self.assertIn("market", all_df.columns)
        self.assertIn("symbol", all_df.columns)

    def test_delete_stock_features(self):
        """deleteで指定銘柄のデータが削除されることを確認"""
        df = pd.DataFrame({"Close": [100.0], "y": [101.0]})
        db_module.upsert_stock_features("us", "AAPL", df)
        db_module.upsert_stock_features("us", "GOOG", df)

        db_module.delete_stock_features("us", "AAPL")

        self.assertIsNone(db_module.load_stock_features("us", "AAPL"))
        self.assertIsNotNone(db_module.load_stock_features("us", "GOOG"))

    def test_get_all_symbols(self):
        """全銘柄の (market, symbol) リストが取得できることを確認"""
        df = pd.DataFrame({"Close": [100.0], "y": [101.0]})
        db_module.upsert_stock_features("us", "AAPL", df)
        db_module.upsert_stock_features("jp", "7203", df)
        db_module.upsert_stock_features("us", "GOOG", df)

        symbols = db_module.get_all_symbols()
        self.assertEqual(len(symbols), 3)
        self.assertIn(("us", "AAPL"), symbols)
        self.assertIn(("jp", "7203"), symbols)
        self.assertIn(("us", "GOOG"), symbols)

    def test_save_and_load_prediction_results(self):
        """prediction_results の保存と取得が正しく動作することを確認"""
        results = [
            PredictionResult(
                market="us",
                symbol="AAPL",
                current_price=150.0,
                avg_pred_price=155.0,
                diff_ratio=0.033,
                model_count=2,
            ),
            PredictionResult(
                market="us",
                symbol="GOOG",
                current_price=2800.0,
                avg_pred_price=2850.0,
                diff_ratio=0.018,
                model_count=2,
            ),
        ]

        db_module.save_prediction_results("20260228_120000", results)

        loaded = db_module.load_prediction_results("20260228_120000")
        self.assertEqual(len(loaded), 2)

        # top_n/worst_nフィルタ
        loaded_top = db_module.load_prediction_results("20260228_120000", top_n=1)
        self.assertEqual(len(loaded_top), 1)
        self.assertEqual(loaded_top["symbol"].iloc[0], "AAPL")  # diff_ratio降順で1位

        loaded_worst = db_module.load_prediction_results("20260228_120000", worst_n=1)
        self.assertEqual(len(loaded_worst), 1)
        self.assertEqual(loaded_worst["symbol"].iloc[0], "GOOG")  # diff_ratio昇順で1位

    def test_load_latest_prediction_timestamp(self):
        """最新のタイムスタンプが正しく取得できることを確認"""
        result_aapl = [
            PredictionResult(
                market="us",
                symbol="AAPL",
                current_price=150.0,
                avg_pred_price=155.0,
                diff_ratio=0.033,
                model_count=2,
            )
        ]
        result_msft = [
            PredictionResult(
                market="us",
                symbol="MSFT",
                current_price=300.0,
                avg_pred_price=310.0,
                diff_ratio=0.033,
                model_count=2,
            )
        ]

        db_module.save_prediction_results("20260101_100000", result_aapl)
        db_module.save_prediction_results("20260228_120000", result_aapl)
        # 同じ銘柄(AAPL)への2回目のinsertはDelete-Insertで上書き
        db_module.save_prediction_results("20260115_080000", result_msft)

        latest = db_module.load_latest_prediction_timestamp()
        self.assertEqual(latest, "20260228_120000")

    def test_load_prediction_results_latest_default(self):
        """predicted_at=None で最新の結果が取得できることを確認"""
        results_aapl = [
            PredictionResult(
                market="us",
                symbol="AAPL",
                current_price=150.0,
                avg_pred_price=155.0,
                diff_ratio=0.033,
                model_count=2,
            )
        ]
        results_goog = [
            PredictionResult(
                market="us",
                symbol="GOOG",
                current_price=2800.0,
                avg_pred_price=2850.0,
                diff_ratio=0.018,
                model_count=2,
            )
        ]

        db_module.save_prediction_results("20260101_100000", results_aapl)
        db_module.save_prediction_results("20260228_120000", results_goog)

        loaded = db_module.load_prediction_results()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded["symbol"].iloc[0], "GOOG")

    def test_load_prediction_markets(self):
        """マーケット一覧が正しく取得できることを確認"""
        results = [
            PredictionResult(
                market="us",
                symbol="AAPL",
                current_price=150.0,
                avg_pred_price=155.0,
                diff_ratio=0.033,
                model_count=2,
            ),
            PredictionResult(
                market="jp",
                symbol="7203",
                current_price=2000.0,
                avg_pred_price=2050.0,
                diff_ratio=0.025,
                model_count=2,
            ),
        ]

        db_module.save_prediction_results("20260228_120000", results)

        markets = db_module.load_prediction_markets("20260228_120000")
        self.assertEqual(sorted(markets), ["jp", "us"])

    def test_dynamic_column_addition(self):
        """後から新しい列が追加されても正しく保存・取得できることを確認"""
        df1 = pd.DataFrame({"Close": [100.0], "y": [101.0]})
        db_module.upsert_stock_features("us", "AAPL", df1)

        df2 = pd.DataFrame({"Close": [200.0], "y": [201.0], "RSI": [55.0], "MACD": [1.5]})
        db_module.upsert_stock_features("us", "GOOG", df2)

        loaded = db_module.load_stock_features("us", "GOOG")
        self.assertIn("RSI", loaded.columns)
        self.assertIn("MACD", loaded.columns)
        self.assertAlmostEqual(loaded["RSI"].iloc[0], 55.0)


class TestExperimentRunsIntegration(unittest.TestCase):
    """experiment_runs テーブルの統合テスト（実DB使用・R-211）"""

    def setUp(self):
        """各テスト前に一時DBで初期化"""
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_experiment.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        """各テスト後にDBをクリーンアップ"""
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

    def test_save_and_load_experiment_run(self):
        """実DBでsave_experiment_run → load_experiment_runsの往復確認"""
        run_id = "integration-run-001"
        db_module.save_experiment_run(
            run_id=run_id,
            market="jp",
            symbol="7203",
            model_name="StockXGBoostModel",
            trained_at="20260314_120000",
            horizon=1,
            rmse=0.012,
            directional_accuracy=0.58,
            n_samples=500,
            feature_names=["close", "volume", "sma5"],
            params={"learning_rate": 0.05, "n_estimators": 200},
        )

        df = db_module.load_experiment_runs(market="jp", symbol="7203")

        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["run_id"], run_id)
        self.assertEqual(row["market"], "jp")
        self.assertEqual(row["symbol"], "7203")
        self.assertEqual(row["model_name"], "StockXGBoostModel")
        self.assertAlmostEqual(row["rmse"], 0.012)
        self.assertAlmostEqual(row["directional_accuracy"], 0.58)
        self.assertEqual(int(row["n_samples"]), 500)
        self.assertEqual(int(row["n_features"]), 3)
        self.assertEqual(len(str(row["feature_hash"])), 16)

    def test_load_best_run_by_directional_accuracy(self):
        """複数ランを保存してload_best_runが最良ランを正しく返すか確認"""
        runs = [
            ("run-da-low", 0.55),
            ("run-da-high", 0.70),
            ("run-da-mid", 0.62),
        ]
        for i, (run_id, da) in enumerate(runs):
            db_module.save_experiment_run(
                run_id=run_id,
                market="us",
                symbol="AAPL",
                model_name="StockXGBoostModel",
                trained_at=f"2026031{i}_120000",
                directional_accuracy=da,
            )

        best = db_module.load_best_run(
            market="us",
            symbol="AAPL",
            model_name="StockXGBoostModel",
            metric="directional_accuracy",
        )

        self.assertIsNotNone(best)
        self.assertAlmostEqual(best["directional_accuracy"], 0.70)
        self.assertEqual(best["run_id"], "run-da-high")

    def test_load_best_run_returns_none_when_empty(self):
        """空テーブルでload_best_runがNoneを返すか確認"""
        result = db_module.load_best_run(
            market="us",
            symbol="MSFT",
            model_name="StockXGBoostModel",
        )
        self.assertIsNone(result)

    def test_run_id_uniqueness_constraint(self):
        """同一run_idでのINSERT OR REPLACE動作確認（レコードが上書きされること）"""
        run_id = "run-overwrite-test"
        db_module.save_experiment_run(
            run_id=run_id,
            market="jp",
            symbol="6758",
            model_name="StockLightGBMModel",
            trained_at="20260314_120000",
            directional_accuracy=0.55,
        )
        # 同じrun_idで上書き保存（directional_accuracyを更新）
        db_module.save_experiment_run(
            run_id=run_id,
            market="jp",
            symbol="6758",
            model_name="StockLightGBMModel",
            trained_at="20260314_120000",
            directional_accuracy=0.65,
        )

        df = db_module.load_experiment_runs(market="jp", symbol="6758")
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["directional_accuracy"], 0.65)


if __name__ == "__main__":
    unittest.main()
