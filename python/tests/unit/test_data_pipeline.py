"""data_pipeline モジュールのユニットテスト"""

import glob
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.market_data.pipeline import (
    fetch_stock_data_with_features,
    save_features_to_db,
    save_stock_data_with_features,
)


class _TmpDbTestCase(unittest.TestCase):
    """一時DBを使うテストの共通基底クラス（テスト発見対象外）"""

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
        for path in glob.glob(self.tmp_db + "*"):
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_ohlcv(self, periods=60):
        """テスト用OHLCVデータを生成する（ラグ特徴量・テクニカル指標生成に十分な行数）"""
        dates = pd.date_range("2023-01-01", periods=periods, freq="D")
        return pd.DataFrame(
            {
                "Open": np.linspace(100, 120, periods),
                "High": np.linspace(101, 121, periods),
                "Low": np.linspace(99, 119, periods),
                "Close": np.linspace(100, 120, periods),
                "Volume": np.random.randint(1000, 5000, periods),
            },
            index=dates,
        )


class TestFetchStockDataWithFeatures(_TmpDbTestCase):
    """fetch_stock_data_with_features 関数のテスト"""

    @patch("src.market_data.pipeline.save_raw_ohlcv")
    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.should_fetch_fresh_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_full_fetch_returns_features_tuple(
        self, mock_from_db, mock_fresh_check, mock_get_data, mock_save_raw
    ):
        """DBキャッシュなし・新規取得時に(market,symbol,data,raw)タプルが返ることを確認"""
        mock_from_db.return_value = None
        mock_fresh_check.return_value = True
        mock_get_data.return_value = self._make_ohlcv(60)

        result = fetch_stock_data_with_features(
            market="us", symbol="AAPL", start_date="2023-01-01", end_date="2023-03-31"
        )

        self.assertIsNotNone(result)
        market, symbol, data, _raw, _quality = result
        self.assertEqual(market, "us")
        self.assertEqual(symbol, "AAPL")
        self.assertIn("y", data.columns)
        self.assertIn("market_encoded", data.columns)
        self.assertEqual(data["market_encoded"].iloc[0], 0)
        # ラグ特徴量・テクニカル指標によるNaN除去後も有効行数が残っている
        self.assertGreater(len(data), 5)

    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.should_fetch_fresh_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_cache_hit_skips_api_call(self, mock_from_db, mock_fresh_check, mock_get_data):
        """DBキャッシュが最新の場合はAPIを呼び出さずキャッシュデータを使うことを確認"""
        mock_from_db.return_value = self._make_ohlcv(60)
        mock_fresh_check.return_value = False

        result = fetch_stock_data_with_features(
            market="us", symbol="MSFT", start_date="2023-01-01", end_date="2023-03-31"
        )

        mock_get_data.assert_not_called()
        self.assertIsNotNone(result)

    @patch("src.market_data.pipeline.save_raw_ohlcv")
    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.should_fetch_fresh_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_api_returns_none_yields_none(
        self, mock_from_db, mock_fresh_check, mock_get_data, mock_save_raw
    ):
        """APIがNoneを返した場合に結果がNoneであることを確認"""
        mock_from_db.return_value = None
        mock_fresh_check.return_value = True
        mock_get_data.return_value = None

        result = fetch_stock_data_with_features(
            market="us", symbol="NONE", start_date="2023-01-01", end_date="2023-03-31"
        )

        self.assertIsNone(result)


class TestSaveFeaturesDb(_TmpDbTestCase):
    """save_features_to_db 関数のテスト"""

    def _build_feature_df(self, n_rows=10):
        """最小限の特徴量DataFrameを直接構築する"""
        dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")
        df = pd.DataFrame(
            {"close": np.linspace(100, 110, n_rows), "y": np.linspace(1, 2, n_rows)},
            index=dates,
        )
        df["market"] = "us"
        df["symbol"] = "SAVE"
        df["market_encoded"] = 0
        return df

    def test_second_save_overwrites_first_without_duplicates(self):
        """同一銘柄を2回保存しても追記ではなく上書きされることを確認"""
        from src.utils.db import load_stock_features

        data = self._build_feature_df(n_rows=10)
        save_features_to_db("us", "SAVE", data)
        first_count = len(load_stock_features("us", "SAVE"))

        save_features_to_db("us", "SAVE", data)
        second_count = len(load_stock_features("us", "SAVE"))

        self.assertEqual(first_count, second_count)


class TestSaveStockDataWithFeatures(_TmpDbTestCase):
    """save_stock_data_with_features 関数のテスト"""

    @patch("src.market_data.pipeline.get_stock_data")
    def test_saves_features_and_y_column_to_db(self, mock_get_data):
        """株価データからDB保存まで正常に完了し、y列が存在することを確認"""
        mock_get_data.return_value = self._make_ohlcv(60)

        save_stock_data_with_features(
            market="us", symbol="TEST", start_date="2023-01-01", end_date="2023-03-31"
        )

        from src.utils.db import load_stock_features

        result = load_stock_features("us", "TEST")
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIn("y", result.columns)
        # ラグ特徴量・テクニカル指標によるNaN除去後も有効行数が残っている
        self.assertGreater(len(result), 5)

    @patch("src.market_data.pipeline.get_stock_data")
    def test_empty_dataframe_returns_early_without_db_write(self, mock_get_data):
        """空のDataFrameが返された場合、例外なく終了しDBに何も書かれないことを確認"""
        mock_get_data.return_value = pd.DataFrame()

        save_stock_data_with_features(
            market="us", symbol="EMPTY", start_date="2023-01-01", end_date="2023-01-30"
        )

        from src.utils.db import load_stock_features

        result = load_stock_features("us", "EMPTY")
        self.assertTrue(result is None or result.empty)

    @patch("src.market_data.pipeline.get_stock_data")
    def test_none_data_returns_early_without_db_write(self, mock_get_data):
        """Noneが返された場合、例外なく終了しDBに何も書かれないことを確認"""
        mock_get_data.return_value = None

        save_stock_data_with_features(
            market="us", symbol="NONE", start_date="2023-01-01", end_date="2023-01-30"
        )

        from src.utils.db import load_stock_features

        result = load_stock_features("us", "NONE")
        self.assertTrue(result is None or result.empty)

    @patch("src.market_data.pipeline.get_stock_data")
    def test_us_market_encoded_is_zero(self, mock_get_data):
        """US市場のmarket_encoded列が全行0であることを確認"""
        mock_get_data.return_value = self._make_ohlcv(60)

        save_stock_data_with_features(
            market="us", symbol="ENC_US", start_date="2023-01-01", end_date="2023-03-31"
        )

        from src.utils.db import load_stock_features

        result = load_stock_features("us", "ENC_US")
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIn("market_encoded", result.columns)
        self.assertTrue((result["market_encoded"] == 0).all())

    @patch("src.market_data.pipeline.get_stock_data")
    def test_jp_market_encoded_is_one(self, mock_get_data):
        """JP市場のmarket_encoded列が全行1であることを確認"""
        mock_get_data.return_value = self._make_ohlcv(60)

        save_stock_data_with_features(
            market="jp", symbol="7203", start_date="2023-01-01", end_date="2023-03-31"
        )

        from src.utils.db import load_stock_features

        result = load_stock_features("jp", "7203")
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIn("market_encoded", result.columns)
        self.assertTrue((result["market_encoded"] == 1).all())

    @patch("src.market_data.pipeline.get_stock_data")
    def test_rerun_overwrites_without_duplicating_rows(self, mock_get_data):
        """同じ銘柄で再実行すると行数が変わらない（追記なし）ことを確認"""
        mock_get_data.return_value = self._make_ohlcv(60)

        save_stock_data_with_features(
            market="us", symbol="OVR", start_date="2023-01-01", end_date="2023-03-31"
        )
        from src.utils.db import load_stock_features

        count1 = len(load_stock_features("us", "OVR"))

        save_stock_data_with_features(
            market="us", symbol="OVR", start_date="2023-01-01", end_date="2023-03-31"
        )
        count2 = len(load_stock_features("us", "OVR"))

        self.assertEqual(count1, count2)


class TestRunDataBatch(_TmpDbTestCase):
    """run_data_batch 関数のテスト"""

    def _make_phase1_data(self):
        """フェーズ1の成功データを生成するヘルパー"""
        from src.market_data.technical import add_technical_indicators, create_basic_lag_features

        df = self._make_ohlcv(60)
        df = add_technical_indicators(df)
        X, y = create_basic_lag_features(df, n_lags=5, feature_cols=None)
        data = X.copy()
        data["market"] = "us"
        data["symbol"] = "TEST1"
        data["market_encoded"] = 0
        data["y"] = y
        return data

    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.print_summary")
    def test_batch_success_calls_parallel_and_summary(
        self,
        mock_print_summary,
        mock_run_parallel,
        mock_load_symbols,
    ):
        """バッチ処理が正常に完了し、parallel実行とsummary出力が呼ばれることを確認"""
        from src.market_data.pipeline import run_data_batch

        mock_load_symbols.return_value = [
            {"market": "us", "symbol": "TEST1"},
            {"market": "us", "symbol": "TEST2"},
        ]
        data = self._make_phase1_data()
        mock_run_parallel.return_value = [
            {
                "market": "us",
                "symbol": "TEST1",
                "status": "success",
                "data": ("us", "TEST1", data, None, None),
            },
            {
                "market": "us",
                "symbol": "TEST2",
                "status": "success",
                "data": ("us", "TEST2", data, None, None),
            },
        ]

        run_data_batch()

        mock_run_parallel.assert_called_once()
        mock_print_summary.assert_called_once()

    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.watchlist.batch_runner.print_summary")
    def test_batch_no_symbols_skips_processing(
        self,
        mock_print_summary,
        mock_load_symbols,
    ):
        """対象銘柄が空の場合、処理をスキップしsummaryが呼ばれないことを確認"""
        from src.market_data.pipeline import run_data_batch

        mock_load_symbols.return_value = []

        run_data_batch()

        mock_print_summary.assert_not_called()

    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.print_summary")
    def test_batch_with_fetch_errors_still_calls_summary(
        self,
        mock_print_summary,
        mock_run_parallel,
        mock_load_symbols,
    ):
        """フェーズ1でエラーが発生しても最終サマリーが出力されることを確認"""
        from src.market_data.pipeline import run_data_batch

        mock_load_symbols.return_value = [
            {"market": "us", "symbol": "TEST1"},
            {"market": "us", "symbol": "TEST2"},
        ]
        data = self._make_phase1_data()
        mock_run_parallel.return_value = [
            {
                "market": "us",
                "symbol": "TEST1",
                "status": "success",
                "data": ("us", "TEST1", data, None, None),
            },
            {"market": "us", "symbol": "TEST2", "status": "error", "error": "取得失敗"},
        ]

        run_data_batch()

        mock_print_summary.assert_called_once()

    @patch("src.watchlist.batch_runner.load_target_symbols")
    @patch("src.watchlist.batch_runner.run_parallel")
    @patch("src.watchlist.batch_runner.print_summary")
    def test_batch_fetch_only_skips_db_write(
        self,
        mock_print_summary,
        mock_run_parallel,
        mock_load_symbols,
    ):
        """fetch_only=TrueではDB書き込みを行わず取得フェーズのsummaryが出力されることを確認"""
        from src.market_data.pipeline import run_data_batch

        mock_load_symbols.return_value = [{"market": "us", "symbol": "TEST1"}]
        data = self._make_phase1_data()
        mock_run_parallel.return_value = [
            {
                "market": "us",
                "symbol": "TEST1",
                "status": "success",
                "data": ("us", "TEST1", data, None),
            },
        ]

        run_data_batch(fetch_only=True)

        mock_print_summary.assert_called_once()
        # 取得フェーズのsummaryラベルに「保存なし」が含まれることを確認
        summary_label = mock_print_summary.call_args[0][0]
        self.assertIn("保存なし", summary_label)


if __name__ == "__main__":
    unittest.main()
