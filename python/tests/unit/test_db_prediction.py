"""db/prediction モジュールのユニットテスト（一時DB使用）"""

import os
import tempfile
import unittest

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.domain.types import PredictionResult, TrainingMetrics
from src.utils.db.prediction import (
    load_latest_prediction_timestamp,
    load_prediction_markets,
    load_prediction_results,
    save_model_metrics,
    save_prediction_results,
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
        for path in (self.tmp_db, self.tmp_db + ".wal"):
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def _make_results(self, market="us", symbol="AAPL", n=1):
        return [
            PredictionResult(
                market=market,
                symbol=symbol,
                current_price=100.0 + i,
                avg_pred_price=102.0 + i,
                diff_ratio=0.02 * (i + 1),
                model_count=2,
            )
            for i in range(n)
        ]


class TestSavePredictionResults(_TmpDbTestCase):
    """save_prediction_results のテスト"""

    def test_save_and_load_roundtrip(self):
        """保存した予測結果を読み込めること"""
        results = self._make_results("us", "AAPL")
        save_prediction_results("20260403_120000", results)

        df = load_prediction_results("20260403_120000")
        self.assertEqual(len(df), 1)
        self.assertEqual(df["symbol"].iloc[0], "AAPL")
        self.assertEqual(df["market"].iloc[0], "us")

    def test_delete_insert_no_duplicate(self):
        """同じmarket/symbolに2回保存しても重複しないこと（Delete-Insert）"""
        results = self._make_results("us", "AAPL")
        save_prediction_results("20260403_120000", results)
        save_prediction_results("20260403_120000", results)

        df = load_prediction_results("20260403_120000")
        self.assertEqual(len(df), 1)

    def test_multi_symbols_saved(self):
        """複数銘柄が正しく保存されること"""
        results = [
            PredictionResult("us", "AAPL", 150.0, 153.0, 0.02, 2),
            PredictionResult("us", "MSFT", 300.0, 306.0, 0.02, 2),
            PredictionResult("jp", "7203", 3000.0, 3060.0, 0.02, 2),
        ]
        save_prediction_results("20260403_130000", results)

        df = load_prediction_results("20260403_130000")
        self.assertEqual(len(df), 3)

    def test_predicted_at_stored(self):
        """predicted_at が正しく保存されること"""
        results = self._make_results()
        save_prediction_results("20260403_150000", results)

        df = load_prediction_results("20260403_150000")
        self.assertTrue((df["predicted_at"] == "20260403_150000").all())

    def test_diff_ratio_stored(self):
        """diff_ratio が正しく保存されること"""
        results = [PredictionResult("us", "AAPL", 100.0, 103.0, 0.03, 2)]
        save_prediction_results("20260403_120000", results)

        df = load_prediction_results("20260403_120000")
        self.assertAlmostEqual(df["diff_ratio"].iloc[0], 0.03, places=5)


class TestLoadLatestPredictionTimestamp(_TmpDbTestCase):
    """load_latest_prediction_timestamp のテスト"""

    def test_returns_none_when_empty(self):
        """データがない場合はNoneを返す"""
        result = load_latest_prediction_timestamp()
        self.assertIsNone(result)

    def test_returns_latest_of_multiple(self):
        """複数タイムスタンプがある場合は最新を返す"""
        # Delete-Insert方式のため、同じmarket/symbolは上書きされる。
        # 銘柄を分けて各タイムスタンプのレコードを保持する。
        save_prediction_results(
            "20260401_120000", [PredictionResult("us", "AAPL", 100.0, 101.0, 0.01, 2)]
        )
        save_prediction_results(
            "20260402_120000", [PredictionResult("us", "MSFT", 200.0, 202.0, 0.01, 2)]
        )
        save_prediction_results(
            "20260403_120000", [PredictionResult("us", "GOOG", 150.0, 151.5, 0.01, 2)]
        )

        ts = load_latest_prediction_timestamp()
        self.assertEqual(ts, "20260403_120000")

    def test_returns_single_timestamp(self):
        """1件のみの場合はそのタイムスタンプを返す"""
        results = self._make_results()
        save_prediction_results("20260403_090000", results)

        ts = load_latest_prediction_timestamp()
        self.assertEqual(ts, "20260403_090000")


class TestLoadPredictionResults(_TmpDbTestCase):
    """load_prediction_results のテスト"""

    def setUp(self):
        super().setUp()
        self.ts = "20260403_120000"
        results = [
            PredictionResult("us", "AAPL", 150.0, 156.0, 0.04, 2),
            PredictionResult("us", "MSFT", 300.0, 309.0, 0.03, 2),
            PredictionResult("us", "GOOG", 200.0, 198.0, -0.01, 2),
            PredictionResult("jp", "7203", 3000.0, 3090.0, 0.03, 2),
            PredictionResult("jp", "9984", 9000.0, 9270.0, 0.03, 2),
        ]
        save_prediction_results(self.ts, results)

    def test_load_all_records(self):
        """全件取得できること"""
        df = load_prediction_results(self.ts)
        self.assertEqual(len(df), 5)

    def test_top_n_returns_descending(self):
        """top_n指定で上位N件を差異率降順で取得できること"""
        df = load_prediction_results(self.ts, top_n=2)
        self.assertEqual(len(df), 2)
        self.assertGreaterEqual(df["diff_ratio"].iloc[0], df["diff_ratio"].iloc[1])

    def test_worst_n_returns_ascending(self):
        """worst_n指定で下位N件を差異率昇順で取得できること"""
        df = load_prediction_results(self.ts, worst_n=2)
        self.assertEqual(len(df), 2)
        self.assertLessEqual(df["diff_ratio"].iloc[0], df["diff_ratio"].iloc[1])

    def test_market_filter_us(self):
        """market=us フィルタが機能すること"""
        df = load_prediction_results(self.ts, market="us")
        self.assertTrue((df["market"] == "us").all())
        self.assertEqual(len(df), 3)

    def test_market_filter_jp(self):
        """market=jp フィルタが機能すること"""
        df = load_prediction_results(self.ts, market="jp")
        self.assertTrue((df["market"] == "jp").all())
        self.assertEqual(len(df), 2)

    def test_limit_without_timestamp(self):
        """limit指定・predicted_at=None で全タイムスタンプから件数制限取得できること"""
        df = load_prediction_results(predicted_at=None, limit=3)
        self.assertLessEqual(len(df), 3)

    def test_returns_empty_for_unknown_timestamp(self):
        """存在しないタイムスタンプでは空DataFrameを返すこと"""
        df = load_prediction_results("99990101_000000")
        self.assertTrue(df.empty)

    def test_auto_resolve_latest_timestamp(self):
        """predicted_at=None のとき最新タイムスタンプを自動解決すること"""
        df = load_prediction_results()
        self.assertFalse(df.empty)
        self.assertTrue((df["predicted_at"] == self.ts).all())


class TestLoadPredictionMarkets(_TmpDbTestCase):
    """load_prediction_markets のテスト"""

    def test_returns_market_list(self):
        """保存したマーケット一覧が返ること"""
        results = [
            PredictionResult("us", "AAPL", 100.0, 101.0, 0.01, 2),
            PredictionResult("jp", "7203", 3000.0, 3030.0, 0.01, 2),
        ]
        save_prediction_results("20260403_120000", results)

        markets = load_prediction_markets("20260403_120000")
        self.assertIn("us", markets)
        self.assertIn("jp", markets)

    def test_returns_empty_for_unknown_timestamp(self):
        """存在しないタイムスタンプでは空リストを返す"""
        markets = load_prediction_markets("99990101_000000")
        self.assertEqual(markets, [])

    def test_returns_empty_when_no_data(self):
        """データがない状態でNone渡しは空リスト"""
        markets = load_prediction_markets(None)
        self.assertEqual(markets, [])

    def test_uses_latest_timestamp_when_none(self):
        """timestamp=None のとき最新タイムスタンプから取得すること"""
        results = [PredictionResult("us", "AAPL", 100.0, 101.0, 0.01, 2)]
        save_prediction_results("20260403_120000", results)

        markets = load_prediction_markets()
        self.assertIn("us", markets)

    def test_sorted_alphabetically(self):
        """マーケット一覧がアルファベット順であること"""
        results = [
            PredictionResult("us", "AAPL", 100.0, 101.0, 0.01, 2),
            PredictionResult("jp", "7203", 3000.0, 3030.0, 0.01, 2),
        ]
        save_prediction_results("20260403_120000", results)

        markets = load_prediction_markets("20260403_120000")
        self.assertEqual(markets, sorted(markets))


class TestSaveModelMetrics(_TmpDbTestCase):
    """save_model_metrics のテスト"""

    def test_save_without_error(self):
        """保存がエラーなく実行されること"""
        metrics = TrainingMetrics(rmse=0.012, directional_accuracy=0.65, n_samples=500)
        save_model_metrics("us", "AAPL", "StockXGBoostModel", "20260403_120000", metrics)

    def test_save_multiple_models(self):
        """複数モデルの指標を保存できること"""
        metrics_xgb = TrainingMetrics(rmse=0.010, directional_accuracy=0.62, n_samples=500)
        metrics_lgb = TrainingMetrics(rmse=0.011, directional_accuracy=0.63, n_samples=500)
        save_model_metrics("us", "AAPL", "StockXGBoostModel", "20260403_120000", metrics_xgb)
        save_model_metrics("us", "AAPL", "StockLightGBMModel", "20260403_120000", metrics_lgb)

    def test_save_for_jp_market(self):
        """JP市場の指標も保存できること"""
        metrics = TrainingMetrics(rmse=0.015, directional_accuracy=0.60, n_samples=300)
        save_model_metrics("jp", "7203", "StockXGBoostModel", "20260403_120000", metrics)


if __name__ == "__main__":
    unittest.main()
