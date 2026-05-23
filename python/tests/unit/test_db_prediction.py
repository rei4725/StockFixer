"""db/prediction モジュールのユニットテスト（一時DB使用）"""

import glob
import os
import shutil
import tempfile
import unittest

import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.prediction.types import PredictionResult, TrainingMetrics
from src.prediction.db import (
    load_drift_summary,
    load_excluded_features,
    load_latest_prediction_timestamp,
    load_open_close_advantage_summary,
    load_paper_real_diff_summary,
    load_prediction_accuracy,
    load_prediction_markets,
    load_prediction_results,
    load_shap_latest,
    load_weekly_accuracy_snapshots,
    save_feature_selection,
    save_model_metrics,
    save_prediction_accuracy,
    save_prediction_results,
    save_shap_values,
    save_weekly_accuracy_snapshot,
    upsert_paper_real_diff,
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


class TestFeatureSelection(_TmpDbTestCase):
    def test_save_and_load_excluded_features(self):
        selection_df = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "importance_mean": [0.10, 0.03, -0.01],
                "importance_std": [0.01, 0.01, 0.02],
                "importance_rank": [1, 2, 3],
                "is_excluded": [False, False, True],
                "protected_by_shap": [False, False, False],
            }
        )
        save_feature_selection("us", "AAPL", "StockXGBoostModel", "20260405_120000", selection_df)
        save_feature_selection("us", "AAPL", "StockLightGBMModel", "20260405_120000", selection_df)

        excluded = load_excluded_features("us", "AAPL")

        self.assertEqual(excluded, ["f3"])

    def test_protected_feature_is_not_excluded(self):
        selection_df = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "importance_mean": [0.03, -0.01],
                "importance_std": [0.01, 0.02],
                "importance_rank": [1, 2],
                "is_excluded": [False, True],
                "protected_by_shap": [False, True],
            }
        )
        save_feature_selection("us", "AAPL", "StockXGBoostModel", "20260405_120000", selection_df)
        save_feature_selection("us", "AAPL", "StockLightGBMModel", "20260405_120000", selection_df)

        excluded = load_excluded_features("us", "AAPL")

        self.assertEqual(excluded, [])


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


class TestSavePredictionAccuracy(_TmpDbTestCase):
    """save_prediction_accuracy のテスト"""

    def _make_row(self, **kwargs):
        base = {
            "market": "us",
            "symbol": "AAPL",
            "model_name": "StockXGBoostModel",
            "predicted_at": "20260403_120000",
            "horizon": 1,
            "predicted_price": 102.0,
            "actual_price": 101.5,
            "predicted_ratio": 0.02,
            "actual_ratio": 0.015,
            "direction_match": True,
        }
        base.update(kwargs)
        return base

    def test_save_returns_count(self):
        """保存件数を返すこと"""
        rows = [self._make_row()]
        count = save_prediction_accuracy(rows)
        self.assertEqual(count, 1)

    def test_save_multiple_rows(self):
        """複数行を保存できること"""
        rows = [
            self._make_row(symbol="AAPL"),
            self._make_row(symbol="MSFT"),
            self._make_row(symbol="GOOG"),
        ]
        count = save_prediction_accuracy(rows)
        self.assertEqual(count, 3)

    def test_empty_rows_returns_zero(self):
        """空リストは0を返すこと"""
        count = save_prediction_accuracy([])
        self.assertEqual(count, 0)

    def test_horizon_default_is_one(self):
        """horizonを省略した場合はデフォルト1で保存されること"""
        row = self._make_row()
        row.pop("horizon")
        save_prediction_accuracy([row])
        df = load_prediction_accuracy(horizon=1)
        self.assertFalse(df.empty)


class TestLoadPredictionAccuracy(_TmpDbTestCase):
    """load_prediction_accuracy のテスト"""

    def setUp(self):
        super().setUp()
        rows = [
            {
                "market": "us",
                "symbol": "AAPL",
                "model_name": "XGB",
                "predicted_at": "20260403_120000",
                "horizon": 1,
                "predicted_price": 102.0,
                "actual_price": 101.5,
                "predicted_ratio": 0.02,
                "actual_ratio": 0.015,
                "direction_match": True,
            },
            {
                "market": "us",
                "symbol": "MSFT",
                "model_name": "XGB",
                "predicted_at": "20260403_120000",
                "horizon": 1,
                "predicted_price": 305.0,
                "actual_price": 303.0,
                "predicted_ratio": 0.017,
                "actual_ratio": 0.010,
                "direction_match": True,
            },
            {
                "market": "jp",
                "symbol": "7203",
                "model_name": "XGB",
                "predicted_at": "20260403_120000",
                "horizon": 5,
                "predicted_price": 3100.0,
                "actual_price": 3050.0,
                "predicted_ratio": 0.033,
                "actual_ratio": 0.017,
                "direction_match": True,
            },
        ]
        save_prediction_accuracy(rows)

    def test_load_all_returns_dataframe(self):
        """全件取得でDataFrameが返ること"""
        df = load_prediction_accuracy(horizon=1)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_filter_by_market(self):
        """market フィルタが機能すること"""
        df = load_prediction_accuracy(market="us", horizon=1)
        self.assertTrue((df["market"] == "us").all())

    def test_filter_by_symbol(self):
        """symbol フィルタが機能すること"""
        df = load_prediction_accuracy(symbol="AAPL", horizon=1)
        self.assertTrue((df["symbol"] == "AAPL").all())

    def test_filter_by_horizon(self):
        """horizon フィルタが機能すること"""
        df = load_prediction_accuracy(horizon=5)
        self.assertFalse(df.empty)
        self.assertTrue((df["horizon"] == 5).all())

    def test_limit_parameter(self):
        """limit パラメータが機能すること"""
        df = load_prediction_accuracy(horizon=1, limit=1)
        self.assertLessEqual(len(df), 1)

    def test_empty_for_unknown_horizon(self):
        """存在しないhorizonは空DataFrameを返すこと"""
        df = load_prediction_accuracy(horizon=99)
        self.assertTrue(df.empty)


class TestLoadDriftSummary(_TmpDbTestCase):
    """load_drift_summary のテスト"""

    def setUp(self):
        super().setUp()
        rows = []
        for i in range(5):
            rows.append(
                {
                    "market": "us",
                    "symbol": "AAPL",
                    "model_name": "XGB",
                    "predicted_at": f"2026040{i+1}_120000",
                    "horizon": 1,
                    "predicted_price": 102.0,
                    "actual_price": 101.5,
                    "predicted_ratio": 0.02,
                    "actual_ratio": 0.015,
                    "direction_match": i % 2 == 0,
                }
            )
        rows.append(
            {
                "market": "jp",
                "symbol": "7203",
                "model_name": "XGB",
                "predicted_at": "20260401_120000",
                "horizon": 1,
                "predicted_price": 3100.0,
                "actual_price": 3050.0,
                "predicted_ratio": 0.03,
                "actual_ratio": 0.017,
                "direction_match": True,
            }
        )
        save_prediction_accuracy(rows)

    def test_returns_dataframe(self):
        """DataFrameが返ること"""
        df = load_drift_summary(horizon=1)
        self.assertIsInstance(df, pd.DataFrame)

    def test_contains_direction_accuracy(self):
        """direction_accuracy 列が含まれること"""
        df = load_drift_summary(horizon=1)
        self.assertIn("direction_accuracy", df.columns)

    def test_contains_mean_abs_error(self):
        """mean_abs_error 列が含まれること"""
        df = load_drift_summary(horizon=1)
        self.assertIn("mean_abs_error", df.columns)

    def test_recent_n_limits_records(self):
        """recent_n が機能すること（全件よりも少ないデータで集計）"""
        df_full = load_drift_summary(horizon=1, recent_n=100)
        df_limited = load_drift_summary(horizon=1, recent_n=2)
        self.assertFalse(df_limited.empty)
        # recent_n制限で同じか少ないn_samplesになる
        if len(df_full) > 0 and len(df_limited) > 0:
            self.assertLessEqual(
                df_limited["n_samples"].max(),
                df_full["n_samples"].max(),
            )

    def test_empty_for_unknown_horizon(self):
        """存在しないhorizonは空DataFrameを返すこと"""
        df = load_drift_summary(horizon=99)
        self.assertTrue(df.empty)


class TestSaveShapValues(_TmpDbTestCase):
    """save_shap_values のテスト"""

    def _make_shap_df(self, n=5):
        return pd.DataFrame(
            {
                "feature": [f"feat_{i}" for i in range(n)],
                "shap_mean": [0.1 * i for i in range(n)],
                "shap_rank": list(range(1, n + 1)),
            }
        )

    def test_save_without_error(self):
        """保存がエラーなく実行されること"""
        shap_df = self._make_shap_df()
        save_shap_values("us", "AAPL", "StockXGBoostModel", "20260403_120000", shap_df)

    def test_save_and_load_roundtrip(self):
        """保存したSHAP値を読み込めること"""
        shap_df = self._make_shap_df(3)
        save_shap_values("us", "AAPL", "StockXGBoostModel", "20260403_120000", shap_df)

        result = load_shap_latest("us", "AAPL", "StockXGBoostModel")
        self.assertFalse(result.empty)
        self.assertIn("feature", result.columns)

    def test_delete_insert_no_duplicate(self):
        """同じキーに2回保存しても重複しないこと"""
        shap_df = self._make_shap_df(3)
        save_shap_values("us", "AAPL", "StockXGBoostModel", "20260403_120000", shap_df)
        save_shap_values("us", "AAPL", "StockXGBoostModel", "20260403_120000", shap_df)

        result = load_shap_latest("us", "AAPL", "StockXGBoostModel")
        self.assertEqual(len(result), 3)


class TestLoadShapLatest(_TmpDbTestCase):
    """load_shap_latest のテスト"""

    def setUp(self):
        super().setUp()
        shap_df = pd.DataFrame(
            {
                "feature": [f"feat_{i}" for i in range(10)],
                "shap_mean": [0.1 * (10 - i) for i in range(10)],
                "shap_rank": list(range(1, 11)),
            }
        )
        save_shap_values("us", "AAPL", "StockXGBoostModel", "20260403_120000", shap_df)

    def test_returns_dataframe(self):
        """DataFrameが返ること"""
        result = load_shap_latest("us", "AAPL", "StockXGBoostModel")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_top_n_limit(self):
        """top_n でフィルタした件数が取得されること"""
        result = load_shap_latest("us", "AAPL", "StockXGBoostModel", top_n=3, bottom_n=0)
        self.assertLessEqual(len(result), 3)

    def test_returns_empty_for_unknown_symbol(self):
        """存在しない銘柄は空DataFrameを返すこと"""
        result = load_shap_latest("us", "UNKNOWN", "StockXGBoostModel")
        self.assertTrue(result.empty)

    def test_contains_expected_columns(self):
        """必要な列が含まれること"""
        result = load_shap_latest("us", "AAPL", "StockXGBoostModel")
        for col in ["feature", "shap_mean", "shap_rank", "trained_at"]:
            self.assertIn(col, result.columns)


class TestPaperRealDiff(_TmpDbTestCase):
    def test_upsert_paper_then_live_populates_diff(self):
        upsert_paper_real_diff(
            market="jp",
            symbol="7203",
            predicted_at="20260405_085000",
            side=1,
            signal_price=1000.0,
            mode="paper",
            order_id="paper-1",
            actual_price=1005.0,
        )
        upsert_paper_real_diff(
            market="jp",
            symbol="7203",
            predicted_at="20260405_085000",
            side=1,
            signal_price=1000.0,
            mode="live",
            order_id="real-1",
            actual_price=1008.0,
        )

        with db_module._db_connection() as con:
            row = con.execute(
                "SELECT paper_price, real_price, price_diff, paper_slippage, real_slippage "
                "FROM paper_real_diff WHERE market='jp' AND symbol='7203'"
            ).fetchone()

        self.assertEqual(row[0], 1005.0)
        self.assertEqual(row[1], 1008.0)
        self.assertAlmostEqual(row[2], 3.0)
        self.assertAlmostEqual(row[3], 0.005)
        self.assertAlmostEqual(row[4], 0.008)

    def test_load_paper_real_diff_summary_returns_aggregates(self):
        upsert_paper_real_diff(
            market="jp",
            symbol="7203",
            predicted_at="20260405_085000",
            side=1,
            signal_price=1000.0,
            mode="paper",
            order_id="paper-1",
            actual_price=1005.0,
        )
        upsert_paper_real_diff(
            market="jp",
            symbol="7203",
            predicted_at="20260405_085000",
            side=1,
            signal_price=1000.0,
            mode="live",
            order_id="real-1",
            actual_price=995.0,
        )

        summary = load_paper_real_diff_summary(recent_days=30)

        self.assertEqual(summary["tracked_count"], 1)
        self.assertEqual(summary["comparable_count"], 1)
        self.assertAlmostEqual(summary["avg_abs_price_diff"], 10.0)
        self.assertAlmostEqual(summary["avg_abs_diff_ratio"], 0.01)

    def test_load_open_close_advantage_summary_groups_by_session(self):
        """order_session 別にスリップ集計が行われること"""
        upsert_paper_real_diff(
            market="jp",
            symbol="7203",
            predicted_at="20260405_085000",
            side=1,
            signal_price=1000.0,
            mode="paper",
            order_id="paper-open",
            actual_price=1002.0,
            order_session="open",
        )
        upsert_paper_real_diff(
            market="jp",
            symbol="7204",
            predicted_at="20260405_085000",
            side=1,
            signal_price=2000.0,
            mode="paper",
            order_id="paper-close",
            actual_price=2010.0,
            order_session="close",
        )

        result = load_open_close_advantage_summary(recent_days=30)

        self.assertIn("open", result)
        self.assertIn("close", result)
        self.assertEqual(result["open"]["count"], 1)
        self.assertEqual(result["close"]["count"], 1)
        self.assertAlmostEqual(result["open"]["avg_slippage"], 0.002)
        self.assertAlmostEqual(result["close"]["avg_slippage"], 0.005)

    def test_load_open_close_advantage_summary_empty_when_no_fills(self):
        """約定なしのとき空辞書を返すこと"""
        result = load_open_close_advantage_summary(recent_days=30)
        self.assertEqual(result, {})


class TestWeeklyAccuracySnapshot(_TmpDbTestCase):
    """save_weekly_accuracy_snapshot / load_weekly_accuracy_snapshots のテスト"""

    def _make_accuracy_df(self, market="us", symbol="AAPL", direction_accuracy=0.6, n=10):
        return pd.DataFrame(
            {
                "market": [market],
                "symbol": [symbol],
                "direction_accuracy": [direction_accuracy],
                "mean_abs_error": [0.02],
                "n_samples": [n],
            }
        )

    def test_save_and_load_roundtrip(self):
        """保存した週次スナップショットを読み込めること"""
        df = self._make_accuracy_df()
        save_weekly_accuracy_snapshot("2026-05-11", df)

        result = load_weekly_accuracy_snapshots(n_weeks=4)
        self.assertFalse(result.empty)
        self.assertEqual(result.iloc[0]["week_start"], "2026-05-11")
        self.assertEqual(result.iloc[0]["symbol"], "AAPL")
        self.assertAlmostEqual(result.iloc[0]["direction_accuracy"], 0.6)

    def test_upsert_overwrites_same_week(self):
        """同一 week_start への再保存で既存データが上書きされること"""
        df1 = self._make_accuracy_df(direction_accuracy=0.5)
        df2 = self._make_accuracy_df(direction_accuracy=0.7)
        save_weekly_accuracy_snapshot("2026-05-11", df1)
        save_weekly_accuracy_snapshot("2026-05-11", df2)

        result = load_weekly_accuracy_snapshots(n_weeks=4)
        rows = result[result["week_start"] == "2026-05-11"]
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows.iloc[0]["direction_accuracy"], 0.7)

    def test_n_weeks_limit(self):
        """n_weeks で取得週数が制限されること"""
        for i in range(5):
            week = f"2026-0{i + 1}-02"
            save_weekly_accuracy_snapshot(week, self._make_accuracy_df())

        result = load_weekly_accuracy_snapshots(n_weeks=3)
        self.assertLessEqual(result["week_start"].nunique(), 3)

    def test_empty_df_is_noop(self):
        """空の DataFrame を渡しても例外にならないこと"""
        save_weekly_accuracy_snapshot("2026-05-11", pd.DataFrame())
        result = load_weekly_accuracy_snapshots(n_weeks=4)
        self.assertTrue(result.empty)

    def test_load_returns_empty_when_no_data(self):
        """データがないとき空 DataFrame を返すこと"""
        result = load_weekly_accuracy_snapshots(n_weeks=4)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
