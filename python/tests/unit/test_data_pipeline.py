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

    @patch("src.market_data.pipeline.fetch_pageview_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None)
    @patch("src.market_data.pipeline.fetch_additional_macro_features")
    @patch("src.market_data.pipeline.fetch_cross_asset_features")
    @patch("src.market_data.pipeline.should_fetch_fresh_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_leading_nan_not_backfilled_from_future(
        self,
        mock_from_db,
        mock_fresh_check,
        mock_cross_asset,
        mock_macro,
        _mock_sentiment,
        _mock_pageview,
    ):
        """先頭に欠けのある列が後ろの値で埋められず、その行が落ちることを確認（ルックアヘッド防止）"""
        ohlcv = self._make_ohlcv(120)
        mock_from_db.return_value = ohlcv
        mock_fresh_check.return_value = False
        vix_start = ohlcv.index[50]
        sp500_start = ohlcv.index[55]
        vix = pd.DataFrame({"vix_close": np.arange(70, dtype=float) + 10.0}, index=ohlcv.index[50:])
        sp500 = pd.DataFrame(
            {"sp500_close": np.arange(65, dtype=float) + 4000.0}, index=ohlcv.index[55:]
        )
        mock_cross_asset.return_value = vix
        mock_macro.return_value = sp500

        result = fetch_stock_data_with_features(
            market="us", symbol="AAPL", start_date="2023-01-01", end_date="2023-04-30"
        )

        self.assertIsNotNone(result)
        _market, _symbol, data, _raw, _quality = result
        self.assertGreater(len(data), 0)
        one_day = pd.Timedelta(days=1)
        # 最古のラグ（lag10）まで実データの日付から作られている
        self.assertGreaterEqual(data.index.min() - 10 * one_day, sp500_start)
        self.assertGreaterEqual(data.index.min() - 10 * one_day, vix_start)
        # 各行のラグ値は、その行より前の日付の実データと一致する
        for ts, val in data["sp500_close_lag1"].items():
            self.assertEqual(val, sp500.loc[ts - one_day, "sp500_close"])
        for ts, val in data["vix_close_lag1"].items():
            self.assertEqual(val, vix.loc[ts - one_day, "vix_close"])

    @patch("src.market_data.pipeline.fetch_pageview_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None)
    @patch("src.market_data.pipeline.fetch_additional_macro_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None)
    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.should_fetch_fresh_data", return_value=True)
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_diff_fetch_with_corporate_action_columns_keeps_db_rows(
        self,
        mock_from_db,
        _mock_fresh_check,
        mock_get_data,
        _mock_cross_asset,
        _mock_macro,
        _mock_sentiment,
        _mock_pageview,
    ):
        """差分の足にだけ Dividends / Stock Splits があっても、DB の足が落ちないことを確認

        DB の足はこの2列を保存しないため、合わせると古い行が NaN になる。
        出来事の無い日は 0 なので 0 で埋め、dropna で古い行が消えないようにする。
        """
        ohlcv = self._make_ohlcv(66)
        mock_from_db.return_value = ohlcv.iloc[:60]
        fresh = ohlcv.iloc[60:].copy()
        fresh["Dividends"] = 0.0
        fresh["Stock Splits"] = 0.0
        mock_get_data.return_value = fresh

        result = fetch_stock_data_with_features(
            market="jp",
            symbol="1332",
            start_date="2023-01-01",
            end_date="2023-03-07",
            defer_raw_save=True,
        )

        self.assertIsNotNone(result)
        _market, _symbol, data, raw, _quality = result
        # 新しい足は保存のために返る
        self.assertEqual(len(raw), 6)
        # DB の足から作った行が残っている（新しい 6 行だけにならない）
        self.assertGreater(len(data), 6)
        self.assertLess(data.index.min(), fresh.index.min())
        self.assertNotIn("Dividends_lag1", data.columns)

    @patch("src.market_data.pipeline.fetch_pageview_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None)
    @patch("src.market_data.pipeline.fetch_additional_macro_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None)
    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.should_fetch_fresh_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_feature_columns_same_on_every_fetch_path(
        self,
        mock_from_db,
        mock_fresh_check,
        mock_get_data,
        _mock_cross_asset,
        _mock_macro,
        _mock_sentiment,
        _mock_pageview,
    ):
        """DB で足りる・差分が空・差分を取れた・初めての取得の4つの道で、列の組が同じになる

        yfinance の足だけが Dividends / Stock Splits を持ち、DB の足は持たない。
        道によって列の組が変わると、stock_features に銘柄ごとに全行 NULL の列ができ、
        統合の学習でその銘柄の行が全部落ちる（#816）。
        """
        ohlcv = self._make_ohlcv(66)
        with_actions = ohlcv.copy()
        with_actions["Dividends"] = 0.0
        with_actions["Stock Splits"] = 0.0
        paths = {
            "db_enough": (ohlcv, False, None),
            "diff_empty": (ohlcv, True, pd.DataFrame()),
            "diff_fetched": (ohlcv.iloc[:60], True, with_actions.iloc[60:]),
            "first_fetch": (None, True, with_actions),
        }

        columns = {}
        for name, (from_db, fresh, fetched) in paths.items():
            mock_from_db.return_value = from_db
            mock_fresh_check.return_value = fresh
            mock_get_data.return_value = fetched
            result = fetch_stock_data_with_features(
                market="jp",
                symbol="5202",
                start_date="2023-01-01",
                end_date="2023-03-07",
                defer_raw_save=True,
            )
            self.assertIsNotNone(result, name)
            columns[name] = sorted(result[2].columns)

        for name, cols in columns.items():
            self.assertEqual(cols, columns["db_enough"], name)

    @patch("src.market_data.pipeline.fetch_pageview_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None)
    @patch("src.market_data.pipeline.fetch_additional_macro_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None)
    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.should_fetch_fresh_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_diff_fetch_with_capital_gains_keeps_db_rows_and_columns(
        self,
        mock_from_db,
        mock_fresh_check,
        mock_get_data,
        _mock_cross_asset,
        _mock_macro,
        _mock_sentiment,
        _mock_pageview,
    ):
        """ETF の差分の足にだけ Capital Gains があっても、DB の足が落ちず列の組も同じになる

        yfinance は ETF（jp/1570 など）に Capital Gains を返すが、DB には保存しない。
        合わせると DB の行だけ NaN になり、dropna で古い行が全部消えていた（#839）。
        """
        ohlcv = self._make_ohlcv(66)
        fresh = ohlcv.iloc[60:].copy()
        fresh["Dividends"] = 0.0
        fresh["Stock Splits"] = 0.0
        fresh["Capital Gains"] = 0.0

        mock_from_db.return_value = ohlcv
        mock_fresh_check.return_value = False
        db_only = fetch_stock_data_with_features(
            market="jp",
            symbol="1570",
            start_date="2023-01-01",
            end_date="2023-03-07",
            defer_raw_save=True,
        )

        mock_from_db.return_value = ohlcv.iloc[:60]
        mock_fresh_check.return_value = True
        mock_get_data.return_value = fresh
        diff = fetch_stock_data_with_features(
            market="jp",
            symbol="1570",
            start_date="2023-01-01",
            end_date="2023-03-07",
            defer_raw_save=True,
        )

        self.assertIsNotNone(db_only)
        self.assertIsNotNone(diff)
        data = diff[2]
        # DB の足から作った行が残っている（新しい 6 行だけにならない）
        self.assertGreater(len(data), 6)
        self.assertLess(data.index.min(), fresh.index.min())
        self.assertEqual(sorted(data.columns), sorted(db_only[2].columns))
        self.assertNotIn("Capital_Gains_lag1", data.columns)

    def _make_ohlcv_until(self, last_date: str, periods: int = 60) -> pd.DataFrame:
        """last_date で終わる平日の足を作る"""
        df = self._make_ohlcv(periods)
        df.index = pd.bdate_range(end=last_date, periods=periods)
        return df

    @patch("src.market_data.pipeline.fetch_pageview_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None)
    @patch("src.market_data.pipeline.fetch_additional_macro_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None)
    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_stale_db_bars_with_empty_diff_are_skipped(
        self,
        mock_from_db,
        mock_get_data,
        _mock_cross_asset,
        _mock_macro,
        _mock_sentiment,
        _mock_pageview,
    ):
        """差分が空で DB の最新の足が平日で N 日を超えて古ければ、特徴量を作らずスキップする（#815）"""
        mock_from_db.return_value = self._make_ohlcv_until("2026-09-30")
        mock_get_data.return_value = pd.DataFrame()

        with self.assertLogs("src.market_data.pipeline", level="WARNING") as logs:
            result = fetch_stock_data_with_features(
                market="jp", symbol="5202", start_date="2021-10-20", end_date="2026-10-20"
            )

        self.assertIsNone(result)
        stale_logs = [m for m in logs.output if "2026-09-30" in m]
        self.assertEqual(len(stale_logs), 1)
        self.assertIn("jp/5202", stale_logs[0])

    @patch("src.market_data.pipeline.fetch_pageview_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None)
    @patch("src.market_data.pipeline.fetch_additional_macro_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None)
    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_short_gap_with_empty_diff_still_builds_features(
        self,
        mock_from_db,
        mock_get_data,
        _mock_cross_asset,
        _mock_macro,
        _mock_sentiment,
        _mock_pageview,
    ):
        """祝日明けのような1〜2平日の空きなら、差分が空でも今までどおり特徴量を作る（#815）"""
        # 金曜 10-09 の足まで。月曜 10-12 は祝日、火曜 10-13 の朝に回す
        mock_from_db.return_value = self._make_ohlcv_until("2026-10-09")
        mock_get_data.return_value = pd.DataFrame()

        result = fetch_stock_data_with_features(
            market="jp", symbol="7203", start_date="2021-10-13", end_date="2026-10-13"
        )

        self.assertIsNotNone(result)


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


class TestRunBatchPipeline(_TmpDbTestCase):
    """run_batch_pipeline 関数のテスト"""

    def _make_tasks(self):
        from src.domain.types import SymbolTask

        return [SymbolTask(market="us", symbol="TEST1"), SymbolTask(market="us", symbol="TEST2")]

    @patch("src.market_data.pipeline.save_features_to_db")
    @patch("src.market_data.pipeline.fetch_stock_data_with_features")
    def test_batch_success_calls_fetch_and_save(self, mock_fetch, mock_save):
        """バッチ処理が正常に完了し、fetch と save が呼ばれることを確認"""
        from src.market_data.pipeline import run_batch_pipeline

        mock_fetch.return_value = ("us", "TEST1", pd.DataFrame({"x": [1]}), None, None)
        tasks = self._make_tasks()[:1]

        result = run_batch_pipeline(tasks)

        mock_fetch.assert_called_once()
        mock_save.assert_called_once()
        self.assertEqual(len(result.succeeded), 1)

    def test_batch_empty_tasks_returns_empty_result(self):
        """対象銘柄が空の場合、空の BatchResult が返ること"""
        from src.market_data.pipeline import run_batch_pipeline

        result = run_batch_pipeline([])

        self.assertEqual(len(result.succeeded), 0)
        self.assertEqual(len(result.failed), 0)

    @patch("src.market_data.pipeline.save_features_to_db")
    @patch("src.market_data.pipeline.fetch_stock_data_with_features")
    def test_batch_fetch_error_becomes_failed(self, mock_fetch, mock_save):
        """fetch でエラーが発生した場合、failed に記録されること"""
        from src.market_data.pipeline import run_batch_pipeline

        mock_fetch.side_effect = RuntimeError("fetch 失敗")

        result = run_batch_pipeline(self._make_tasks()[:1])

        self.assertEqual(len(result.failed), 1)
        mock_save.assert_not_called()

    @patch("src.market_data.pipeline.fetch_stock_data_with_features")
    def test_batch_fetch_only_skips_db_write(self, mock_fetch):
        """fetch_only=True では DB 書き込みをスキップして BatchResult を返すこと"""
        from src.market_data.pipeline import run_batch_pipeline

        mock_fetch.return_value = ("us", "TEST1", pd.DataFrame({"x": [1]}), None, None)

        with patch("src.market_data.pipeline.save_features_to_db") as mock_save:
            result = run_batch_pipeline(self._make_tasks()[:1], fetch_only=True)
            mock_save.assert_not_called()

        self.assertEqual(len(result.succeeded), 1)

    @patch("src.market_data.pipeline.save_raw_ohlcv")
    @patch("src.market_data.pipeline.save_features_to_db")
    @patch("src.market_data.pipeline.fetch_pageview_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None)
    @patch("src.market_data.pipeline.fetch_additional_macro_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None)
    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_batch_stale_bars_count_as_skip_without_db_write(
        self,
        mock_from_db,
        mock_get_data,
        _mock_cross_asset,
        _mock_macro,
        _mock_sentiment,
        _mock_pageview,
        mock_save,
        mock_save_raw,
    ):
        """足が止まって差分も空の銘柄はスキップに数え、stock_features を上書きしない（#815）"""
        from src.market_data.pipeline import run_batch_pipeline

        ohlcv = self._make_ohlcv(60)
        ohlcv.index = pd.bdate_range(end="2023-03-01", periods=60)
        mock_from_db.return_value = ohlcv
        mock_get_data.return_value = pd.DataFrame()

        result = run_batch_pipeline(self._make_tasks()[:1])

        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(len(result.succeeded), 0)
        self.assertEqual(len(result.failed), 0)
        mock_save.assert_not_called()
        mock_save_raw.assert_not_called()

    @patch("src.market_data.pipeline.save_raw_ohlcv")
    @patch("src.market_data.pipeline.save_features_to_db")
    @patch("src.market_data.pipeline.fetch_pageview_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_news_sentiment_with_llm", return_value=None)
    @patch("src.market_data.pipeline.fetch_additional_macro_features", return_value=None)
    @patch("src.market_data.pipeline.fetch_cross_asset_features", return_value=None)
    @patch("src.market_data.pipeline.should_fetch_fresh_data", return_value=True)
    @patch("src.market_data.pipeline.get_stock_data")
    @patch("src.market_data.pipeline.get_raw_ohlcv_from_db")
    def test_batch_feature_skip_still_saves_fetched_raw_bars(
        self,
        mock_from_db,
        mock_get_data,
        _mock_fresh_check,
        _mock_cross_asset,
        _mock_macro,
        _mock_sentiment,
        _mock_pageview,
        mock_save,
        mock_save_raw,
    ):
        """行の不足で特徴量をスキップしても、取った差分の足は market_data_raw に保存する（#840）"""
        from src.market_data.pipeline import run_batch_pipeline

        db_bars = self._make_ohlcv(10)
        db_bars.index = pd.bdate_range(end="2023-03-01", periods=10)
        fresh_bars = self._make_ohlcv(5)
        fresh_bars.index = pd.bdate_range(start="2023-03-02", periods=5)
        mock_from_db.return_value = db_bars
        mock_get_data.return_value = fresh_bars

        result = run_batch_pipeline(self._make_tasks()[:1])

        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(len(result.succeeded), 0)
        self.assertEqual(len(result.failed), 0)
        mock_save.assert_not_called()
        mock_save_raw.assert_called_once()
        market, symbol, saved = mock_save_raw.call_args.args
        self.assertEqual((market, symbol), ("us", "TEST1"))
        pd.testing.assert_frame_equal(saved, fresh_bars)


if __name__ == "__main__":
    unittest.main()
