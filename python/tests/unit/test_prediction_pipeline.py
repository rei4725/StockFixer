"""prediction_pipeline モジュールのユニットテスト"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestGetOptimalParams(unittest.TestCase):
    """get_optimal_params のテスト（JSON ファイルをモック）"""

    def test_returns_empty_when_file_not_exists(self):
        from src.prediction.prediction_pipeline import get_optimal_params

        with tempfile.TemporaryDirectory() as tmp:
            # json ファイルを作らない
            nonexistent = Path(tmp) / "config" / "optimal_params.json"
            with patch("src.prediction.prediction_pipeline.Path") as MockPath:
                # chaining: Path(__file__).parents[2] / "config"
                mock_chain = MockPath.return_value
                chained = mock_chain.parents.__getitem__.return_value
                chained.__truediv__.return_value.__truediv__.return_value = nonexistent
                result = get_optimal_params("us", "AAPL")
        self.assertEqual(result, {})

    def test_returns_matching_params(self):
        from src.prediction.prediction_pipeline import get_optimal_params

        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            json_path = config_dir / "optimal_params.json"
            data = {"us_AAPL": {"threshold": 0.02, "metrics": {"sharpe_ratio": 1.5}}}
            json_path.write_text(json.dumps(data), encoding="utf-8")

            with patch("src.prediction.prediction_pipeline.Path") as MockPath:
                parents_mock = MockPath.return_value.parents.__getitem__.return_value
                parents_mock.__truediv__.return_value.__truediv__.return_value = json_path
                result = get_optimal_params("us", "AAPL")
        self.assertEqual(result.get("threshold"), 0.02)

    def test_returns_empty_when_symbol_missing(self):
        from src.prediction.prediction_pipeline import get_optimal_params

        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            json_path = config_dir / "optimal_params.json"
            data = {"us_GOOG": {"threshold": 0.01}}
            json_path.write_text(json.dumps(data), encoding="utf-8")

            with patch("src.prediction.prediction_pipeline.Path") as MockPath:
                parents_mock = MockPath.return_value.parents.__getitem__.return_value
                parents_mock.__truediv__.return_value.__truediv__.return_value = json_path
                result = get_optimal_params("us", "AAPL")
        self.assertEqual(result, {})

    def test_returns_empty_on_invalid_json(self):
        from src.prediction.prediction_pipeline import get_optimal_params

        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            json_path = config_dir / "optimal_params.json"
            json_path.write_text("not-json", encoding="utf-8")

            with patch("src.prediction.prediction_pipeline.Path") as MockPath:
                parents_mock = MockPath.return_value.parents.__getitem__.return_value
                parents_mock.__truediv__.return_value.__truediv__.return_value = json_path
                result = get_optimal_params("us", "AAPL")
        self.assertEqual(result, {})


class TestFindModelFiles(unittest.TestCase):
    """find_model_files のテスト（実ディレクトリを作成して glob を検証）"""

    def _setup_model_dir(self, tmp_dir, entries):
        """entries: [(市場_銘柄, モデル名)] のリスト"""
        for folder, model_name in entries:
            folder_path = os.path.join(tmp_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            open(os.path.join(folder_path, model_name), "w").close()

    def test_finds_matching_models(self):
        from src.prediction.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, [("us_AAPL", "StockXGBoostModel.joblib")])
            result = find_model_files(model_root=tmp)
        self.assertEqual(len(result), 1)
        market, symbol, path = result[0]
        self.assertEqual(market, "us")
        self.assertEqual(symbol, "AAPL")

    def test_returns_empty_when_no_match(self):
        from src.prediction.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, [("us_AAPL", "OtherModel.joblib")])
            result = find_model_files(model_root=tmp, model_name="StockXGBoostModel.joblib")
        self.assertEqual(result, [])

    def test_finds_multiple_symbols(self):
        from src.prediction.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(
                tmp,
                [
                    ("us_AAPL", "StockXGBoostModel.joblib"),
                    ("jp_7203", "StockXGBoostModel.joblib"),
                ],
            )
            result = find_model_files(model_root=tmp)
        self.assertEqual(len(result), 2)
        markets = {r[0] for r in result}
        self.assertIn("us", markets)
        self.assertIn("jp", markets)

    def test_returns_correct_path(self):
        from src.prediction.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, [("us_AAPL", "StockXGBoostModel.joblib")])
            result = find_model_files(model_root=tmp)
            _, _, path = result[0]
            self.assertTrue(os.path.exists(path))

    def test_custom_model_name(self):
        from src.prediction.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, [("us_AAPL", "StockLightGBMModel.joblib")])
            result = find_model_files(model_root=tmp, model_name="StockLightGBMModel.joblib")
        self.assertEqual(len(result), 1)


class TestPredictAllIndividual(unittest.TestCase):
    """predict_all_individual のテスト"""

    @patch("src.prediction.prediction_pipeline.find_model_files")
    @patch("src.prediction.prediction_pipeline.predict_single_stock")
    def test_returns_list_of_prediction_results(self, mock_predict, mock_find):
        """predict_single_stock の結果リストが返ること"""
        from src.prediction.prediction_pipeline import predict_all_individual
        from src.prediction.types import PredictionResult

        mock_find.return_value = [("jp", "7203", "/models/jp_7203/XGB.joblib")]
        mock_predict.return_value = PredictionResult(
            market="jp",
            symbol="7203",
            current_price=1000.0,
            avg_pred_price=1050.0,
            diff_ratio=0.05,
            model_count=2,
        )

        results = predict_all_individual(max_workers=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "7203")

    @patch("src.prediction.prediction_pipeline.find_model_files")
    def test_returns_empty_when_no_models(self, mock_find):
        """モデルが存在しない場合は空リストが返ること"""
        from src.prediction.prediction_pipeline import predict_all_individual

        mock_find.return_value = []

        results = predict_all_individual(max_workers=1)

        self.assertEqual(results, [])

    @patch("src.prediction.prediction_pipeline.find_model_files")
    @patch("src.prediction.prediction_pipeline.predict_single_stock")
    def test_handles_prediction_exception(self, mock_predict, mock_find):
        """予測中に例外が発生しても他の銘柄の処理は継続すること"""
        from src.prediction.prediction_pipeline import predict_all_individual
        from src.prediction.types import PredictionResult

        mock_find.return_value = [
            ("jp", "7203", "/models/jp_7203/XGB.joblib"),
            ("jp", "9984", "/models/jp_9984/XGB.joblib"),
        ]

        def side_effect(market, symbol, model_types):
            if symbol == "7203":
                raise RuntimeError("予測エラー")
            return PredictionResult(
                market="jp",
                symbol="9984",
                current_price=9000.0,
                avg_pred_price=9500.0,
                diff_ratio=0.055,
                model_count=2,
            )

        mock_predict.side_effect = side_effect

        results = predict_all_individual(max_workers=1)

        # 成功した 9984 だけ返ること
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "9984")


class TestOutputTopWorstResults(unittest.TestCase):
    """output_top_worst_results のテスト"""

    @patch("src.prediction.prediction_pipeline.save_prediction_results")
    def test_saves_results_to_db(self, mock_save):
        """save_prediction_results が呼ばれること"""
        from src.prediction.prediction_pipeline import output_top_worst_results
        from src.prediction.types import PredictionResult

        results = [
            PredictionResult(
                market="jp",
                symbol=f"{i}000",
                current_price=1000.0,
                avg_pred_price=1000.0 * (1 + i * 0.01),
                diff_ratio=i * 0.01,
                model_count=2,
            )
            for i in range(1, 6)
        ]
        output_top_worst_results(results)
        mock_save.assert_called_once()

    @patch("src.prediction.prediction_pipeline.save_prediction_results")
    def test_no_save_when_empty(self, mock_save):
        """空リストの場合は save_prediction_results が呼ばれないこと"""
        from src.prediction.prediction_pipeline import output_top_worst_results

        output_top_worst_results([])
        mock_save.assert_not_called()


class TestPredictAllUnified(unittest.TestCase):
    """predict_all_unified のテスト"""

    @patch("src.prediction.prediction_pipeline.get_active_symbols")
    @patch("src.prediction.predict_unified.preload_models")
    @patch("src.prediction.predict_unified.predict_with_unified_model")
    def test_returns_list_of_prediction_results(self, mock_predict, mock_preload, mock_symbols):
        """統合モデルで全銘柄の予測結果リストが返ること"""
        from src.prediction.prediction_pipeline import predict_all_unified
        from src.prediction.types import PredictionResult

        mock_symbols.return_value = [("jp", "7203")]
        mock_preload.return_value = None
        mock_predict.return_value = PredictionResult(
            market="jp",
            symbol="7203",
            current_price=1000.0,
            avg_pred_price=1050.0,
            diff_ratio=0.05,
            model_count=2,
        )
        results = predict_all_unified()
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "7203")

    @patch("src.prediction.prediction_pipeline.get_active_symbols")
    @patch("src.prediction.predict_unified.preload_models")
    def test_returns_empty_when_no_symbols(self, mock_preload, mock_symbols):
        """銘柄がない場合は空リストが返ること"""
        from src.prediction.prediction_pipeline import predict_all_unified

        mock_symbols.return_value = []
        mock_preload.return_value = None
        results = predict_all_unified()
        self.assertEqual(results, [])

    @patch("src.prediction.prediction_pipeline.get_active_symbols")
    @patch("src.prediction.predict_unified.preload_models")
    @patch("src.prediction.predict_unified.predict_with_unified_model")
    def test_predict_exception_returns_empty(self, mock_predict, mock_preload, mock_symbols):
        """予測中に例外が発生しても空リストが返ること（例外が伝播しない）"""
        from src.prediction.prediction_pipeline import predict_all_unified

        mock_symbols.return_value = [("jp", "7203")]
        mock_preload.return_value = None
        mock_predict.side_effect = RuntimeError("予測エラー")
        results = predict_all_unified()
        self.assertEqual(results, [])


class TestRunAccuracyCheck(unittest.TestCase):
    """run_accuracy_check のテスト"""

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_saves_accuracy_to_db(self, mock_load, mock_ohlcv, mock_save, mock_drift):
        """精度データが DB に保存されること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        # predicted_at は YYYYMMDD_HHMMSS 形式
        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "predicted_at": ["20260416_120000"],
                "avg_pred_price": [1050.0],
                "current_price": [1000.0],
            }
        )
        # ohlcv の index は pred_date (2026-04-16) より後の日付が必要
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [1040.0]},
            index=pd.to_datetime(["2026-04-17"]),
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)
        mock_save.assert_called_once()

    @patch("src.prediction.db.load_prediction_results")
    def test_returns_empty_on_no_past_predictions(self, mock_load):
        """過去予測がない場合は正常終了して空 DataFrame が返ること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame()
        result = run_accuracy_check(horizon=1)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_skips_when_ohlcv_empty(self, mock_load, mock_ohlcv, mock_save, mock_drift):
        """OHLCV が空の場合は save_prediction_accuracy が呼ばれないこと"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "predicted_at": ["20260416_120000"],
                "avg_pred_price": [1050.0],
                "current_price": [1000.0],
            }
        )
        mock_ohlcv.return_value = pd.DataFrame()
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)
        mock_save.assert_not_called()


class TestPredictAllUnifiedMultiHorizon(unittest.TestCase):
    """predict_all_unified_multi_horizon のテスト"""

    @patch("src.prediction.multi_horizon.get_active_symbols")
    @patch("src.prediction.predict_unified.preload_models")
    @patch("src.prediction.predict_unified.predict_with_unified_model")
    def test_multi_horizon_returns_results(self, mock_predict, mock_preload, mock_symbols):
        """複数ホライズン（[1, 7]）で予測結果リストが返ること"""
        from src.prediction.prediction_pipeline import predict_all_unified_multi_horizon
        from src.prediction.types import PredictionResult

        mock_symbols.return_value = [("jp", "7203")]
        mock_preload.return_value = None

        def _make_result(market, symbol, **_kwargs):
            return PredictionResult(
                market=market,
                symbol=symbol,
                current_price=1000.0,
                avg_pred_price=1050.0,
                diff_ratio=0.05,
                model_count=2,
            )

        mock_predict.side_effect = _make_result

        results = predict_all_unified_multi_horizon(horizons=[1, 7])

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "7203")

    @patch("src.prediction.multi_horizon.get_active_symbols")
    @patch("src.prediction.predict_unified.preload_models")
    def test_empty_symbols_returns_empty(self, mock_preload, mock_symbols):
        """銘柄リストが空の場合は空リストが返ること"""
        from src.prediction.prediction_pipeline import predict_all_unified_multi_horizon

        mock_symbols.return_value = []
        mock_preload.return_value = None

        results = predict_all_unified_multi_horizon(horizons=[1, 3])
        self.assertEqual(results, [])

    @patch("src.prediction.multi_horizon.get_active_symbols")
    @patch("src.prediction.predict_unified.preload_models")
    @patch("src.prediction.predict_unified.predict_with_unified_model")
    def test_single_horizon_works(self, mock_predict, mock_preload, mock_symbols):
        """ホライズンが1つだけでも正常動作すること"""
        from src.prediction.prediction_pipeline import predict_all_unified_multi_horizon
        from src.prediction.types import PredictionResult

        mock_symbols.return_value = [("us", "AAPL")]
        mock_preload.return_value = None
        mock_predict.return_value = PredictionResult(
            market="us",
            symbol="AAPL",
            current_price=200.0,
            avg_pred_price=210.0,
            diff_ratio=0.05,
            model_count=2,
        )

        results = predict_all_unified_multi_horizon(horizons=[1])

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "AAPL")

    @patch("src.prediction.multi_horizon.get_active_symbols")
    @patch("src.prediction.predict_unified.preload_models")
    @patch("src.prediction.predict_unified.predict_with_unified_model")
    def test_exception_skips_symbol(self, mock_predict, mock_preload, mock_symbols):
        """予測中に例外が発生した銘柄はスキップされること"""
        from src.prediction.prediction_pipeline import predict_all_unified_multi_horizon

        mock_symbols.return_value = [("jp", "7203")]
        mock_preload.return_value = None
        mock_predict.side_effect = RuntimeError("モデルが存在しません")

        results = predict_all_unified_multi_horizon(horizons=[1])
        self.assertEqual(results, [])


class TestRunAccuracyCheckAdditional(unittest.TestCase):
    """run_accuracy_check の追加カバレッジ"""

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_skips_row_with_missing_market(self, mock_load, mock_ohlcv, mock_save, mock_drift):
        """market が None の行はスキップされること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": [None],
                "symbol": ["7203"],
                "predicted_at": ["20260416_120000"],
                "avg_pred_price": [1050.0],
                "current_price": [1000.0],
            }
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)
        mock_save.assert_not_called()

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_skips_row_with_none_pred_price(self, mock_load, mock_ohlcv, mock_save, mock_drift):
        """avg_pred_price が None の行はスキップされること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "predicted_at": ["20260416_120000"],
                "avg_pred_price": [None],
                "current_price": [1000.0],
            }
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)
        mock_save.assert_not_called()

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_skips_when_insufficient_future_data(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """horizon=2 に対して将来データが1件しかない場合はスキップされること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "predicted_at": ["20260416_120000"],
                "avg_pred_price": [1050.0],
                "current_price": [1000.0],
            }
        )
        # pred_date=2026-04-16, future_dates は1件のみ → horizon=2 では不足
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [1040.0]},
            index=pd.to_datetime(["2026-04-17"]),
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=2)
        mock_save.assert_not_called()

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_multiple_predictions_all_saved(self, mock_load, mock_ohlcv, mock_save, mock_drift):
        """複数銘柄の予測が一括で採点・保存されること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp", "jp"],
                "symbol": ["7203", "9984"],
                "predicted_at": ["20260416_120000", "20260416_120000"],
                "avg_pred_price": [1050.0, 9500.0],
                "current_price": [1000.0, 9000.0],
            }
        )
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [1040.0]},
            index=pd.to_datetime(["2026-04-17"]),
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)

        mock_save.assert_called_once()
        saved_rows = mock_save.call_args[0][0]
        self.assertEqual(len(saved_rows), 2)

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_completion_log_counts_skips_by_reason(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """完了ログに、採点を飛ばした件数が理由ごとに出ること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": [None, "jp", "jp", "jp", "jp", "jp", "jp"],
                "symbol": ["1", "bad", "noprice", "noohlcv", "nofuture", "nofuture2", "ok"],
                "predicted_at": [
                    "20260416_120000",
                    "badvalue",
                    "20260416_120000",
                    "20260416_120000",
                    "20260417_223000",
                    "20260417_223000",
                    "20260416_120000",
                ],
                "avg_pred_price": [1050.0, 1050.0, None, 1050.0, 1050.0, 1050.0, 1050.0],
                "current_price": [1000.0] * 7,
            }
        )
        ohlcv = pd.DataFrame({"Close": [1040.0]}, index=pd.to_datetime(["2026-04-17"]))
        mock_ohlcv.side_effect = lambda market, symbol: (
            pd.DataFrame() if symbol == "noohlcv" else ohlcv
        )
        mock_drift.return_value = pd.DataFrame()

        with self.assertLogs("src.prediction.prediction_pipeline", level="INFO") as cm:
            run_accuracy_check(horizon=1)

        done = [m for m in cm.output if "精度チェック完了" in m]
        self.assertEqual(len(done), 1)
        msg = done[0]
        self.assertIn("1件 採点", msg)
        self.assertIn("読み込み=7件", msg)
        self.assertIn("必須の値なし=1", msg)
        self.assertIn("predicted_at形式不正=1", msg)
        self.assertIn("予想値段なし=1", msg)
        self.assertIn("足なし=1", msg)
        self.assertIn("まだデータなし=2", msg)

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_scores_previous_day_prediction_with_next_business_day_bar(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """前の日（UTC 22時台）の予想は、その次の営業日の足で採点されること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        # 本番の日次の道は UTC 22:3x に走り、predicted_at はその UTC の時刻になる
        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp", "us"],
                "symbol": ["7203", "AAPL"],
                "predicted_at": ["20261015_223736", "20261015_223736"],
                "avg_pred_price": [1050.0, 210.0],
                "current_price": [1000.0, 200.0],
            }
        )
        # market_data_raw の ts は TIMESTAMP（tz なし、取引所の日付の 00:00）
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [1000.0, 1040.0]},
            index=pd.DatetimeIndex(pd.to_datetime(["2026-10-15", "2026-10-16"]), name="Date"),
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)

        mock_save.assert_called_once()
        rows = mock_save.call_args[0][0]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["actual_price"], 1040.0)

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_logs_samples_of_no_future_bar_skips(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """まだデータなしで飛ばした行の pred_date・足の最後の日付・足の日付の型が数件ログに出ること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        n = 8
        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp"] * n,
                "symbol": [f"S{i}" for i in range(n)],
                "predicted_at": ["20261016_013000"] * n,
                "avg_pred_price": [1050.0] * n,
                "current_price": [1000.0] * n,
            }
        )
        # 10:30 JST の予想の基準の日は前の日の 10/15。10/16 の足はまだ無い
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [1000.0, 1010.0]},
            index=pd.DatetimeIndex(pd.to_datetime(["2026-10-14", "2026-10-15"]), name="Date"),
        )
        mock_drift.return_value = pd.DataFrame()

        with self.assertLogs("src.prediction.prediction_pipeline", level="INFO") as cm:
            run_accuracy_check(horizon=1)

        samples = [m for m in cm.output if "まだデータなしの例" in m]
        self.assertEqual(len(samples), 1)
        msg = samples[0]
        self.assertIn("jp/S0", msg)
        self.assertIn("predicted_at=20261016_013000", msg)
        self.assertIn("pred_date=2026-10-15", msg)
        self.assertIn("足の最後=2026-10-15 00:00:00", msg)
        self.assertIn("datetime64", msg)
        # 数件に絞る（全件は出さない）
        self.assertNotIn(f"jp/S{n - 1}", msg)
        mock_save.assert_not_called()

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_scores_split_adjusted_raw_bars_on_same_level(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """raw の足が 1:2 の分割で調整し直してあり、保存した現在値が分割前のとき、同じ水準で比べること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "predicted_at": ["20261015_223736"],
                "avg_pred_price": [1050.0],
                "current_price": [1000.0],
            }
        )
        # 分割前の 1000 円は、調整し直した足では 500 円。翌日の実績は調整後 505 円（+1%）
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [500.0, 505.0]},
            index=pd.DatetimeIndex(pd.to_datetime(["2026-10-15", "2026-10-16"]), name="Date"),
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)

        mock_save.assert_called_once()
        row = mock_save.call_args[0][0][0]
        self.assertAlmostEqual(row["actual_ratio"], 0.01)
        self.assertAlmostEqual(row["predicted_ratio"], 0.05)
        self.assertTrue(row["direction_match"])
        self.assertEqual(row["actual_price"], 505.0)
        self.assertAlmostEqual(row["predicted_price"], 525.0)

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_keeps_saved_current_price_without_split(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """分割の無い銘柄は、保存した現在値のまま採点すること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "predicted_at": ["20261015_223736"],
                "avg_pred_price": [1050.0],
                "current_price": [1000.0],
            }
        )
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [990.0, 1040.0]},
            index=pd.DatetimeIndex(pd.to_datetime(["2026-10-15", "2026-10-16"]), name="Date"),
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)

        row = mock_save.call_args[0][0][0]
        self.assertAlmostEqual(row["actual_ratio"], 0.04)
        self.assertAlmostEqual(row["predicted_ratio"], 0.05)
        self.assertEqual(row["predicted_price"], 1050.0)

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_skips_when_price_level_gap_is_not_split_ratio(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """保存した現在値と raw の基準の終値の比が分割の比に合わないときは、採点から外して数えること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "predicted_at": ["20261015_223736"],
                "avg_pred_price": [1050.0],
                "current_price": [1000.0],
            }
        )
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [570.0, 580.0]},
            index=pd.DatetimeIndex(pd.to_datetime(["2026-10-15", "2026-10-16"]), name="Date"),
        )
        mock_drift.return_value = pd.DataFrame()

        with self.assertLogs("src.prediction.prediction_pipeline", level="INFO") as cm:
            run_accuracy_check(horizon=1)

        mock_save.assert_not_called()
        done = [m for m in cm.output if "精度チェック完了" in m]
        self.assertIn("0件 採点", done[0])
        self.assertIn("分割の比が合わない=1", done[0])

    def _score_one(self, market, predicted_at, mock_load, mock_ohlcv, mock_save, mock_drift):
        """10/20〜10/23 の足で1行を採点し、保存した行を返す"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": [market],
                "symbol": ["S"],
                "predicted_at": [predicted_at],
                "avg_pred_price": [1050.0],
                "current_price": [1000.0],
            }
        )
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [990.0, 1000.0, 1020.0, 1040.0]},
            index=pd.DatetimeIndex(
                pd.to_datetime(["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"]),
                name="Date",
            ),
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)

        mock_save.assert_called_once()
        return mock_save.call_args[0][0][0]

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_jp_prediction_during_session_uses_previous_close_as_base(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """日本株で 10:30 JST の補完の予想は、10/21 の終値を基準に 10/22 の終値で採点されること"""
        row = self._score_one("jp", "20261022_013000", mock_load, mock_ohlcv, mock_save, mock_drift)
        self.assertEqual(row["actual_price"], 1020.0)
        self.assertAlmostEqual(row["actual_ratio"], 0.02)

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_us_prediction_after_utc_midnight_uses_previous_close_as_base(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """米国株で 01:30 UTC の補完の予想は、10/21 の終値を基準に 10/22 の終値で採点されること"""
        row = self._score_one("us", "20261022_013000", mock_load, mock_ohlcv, mock_save, mock_drift)
        self.assertEqual(row["actual_price"], 1020.0)
        self.assertAlmostEqual(row["actual_ratio"], 0.02)

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_regular_run_scored_with_next_close(self, mock_load, mock_ohlcv, mock_save, mock_drift):
        """定時の回（22:45 UTC）の予想は、日本株も米国株も 10/22 の終値で採点されること"""
        for market in ("jp", "us"):
            with self.subTest(market=market):
                mock_save.reset_mock()
                row = self._score_one(
                    market, "20261021_224556", mock_load, mock_ohlcv, mock_save, mock_drift
                )
                self.assertEqual(row["actual_price"], 1020.0)
                self.assertAlmostEqual(row["actual_ratio"], 0.02)

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_skips_nan_close_bar_and_scores_with_next_close(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """翌日の足の終値が欠けているとき、その次の終値のある足で採点すること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "predicted_at": ["20261021_224556"],
                "avg_pred_price": [1050.0],
                "current_price": [1000.0],
            }
        )
        mock_ohlcv.return_value = pd.DataFrame(
            {"Close": [1000.0, float("nan"), 1030.0]},
            index=pd.DatetimeIndex(
                pd.to_datetime(["2026-10-21", "2026-10-22", "2026-10-23"]), name="Date"
            ),
        )
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)

        mock_save.assert_called_once()
        row = mock_save.call_args[0][0][0]
        self.assertEqual(row["actual_price"], 1030.0)
        self.assertAlmostEqual(row["actual_ratio"], 0.03)
        self.assertTrue(row["direction_match"])

    @patch("src.prediction.db.load_drift_summary")
    @patch("src.prediction.db.save_prediction_accuracy")
    @patch("src.utils.db.load_raw_ohlcv")
    @patch("src.prediction.db.load_prediction_results")
    def test_nan_base_close_does_not_abort_other_symbols(
        self, mock_load, mock_ohlcv, mock_save, mock_drift
    ):
        """基準の足の終値が欠けた銘柄があっても例外を投げず、ほかの銘柄を採点して保存すること"""
        import pandas as pd

        from src.prediction.prediction_pipeline import run_accuracy_check

        mock_load.return_value = pd.DataFrame(
            {
                "market": ["jp", "jp"],
                "symbol": ["BAD", "GOOD"],
                "predicted_at": ["20261021_224556", "20261021_224556"],
                "avg_pred_price": [1050.0, 1050.0],
                "current_price": [1000.0, 1000.0],
            }
        )
        index = pd.DatetimeIndex(pd.to_datetime(["2026-10-21", "2026-10-22"]), name="Date")
        bars = {
            "BAD": pd.DataFrame({"Close": [float("nan"), 1010.0]}, index=index),
            "GOOD": pd.DataFrame({"Close": [1000.0, 1020.0]}, index=index),
        }
        mock_ohlcv.side_effect = lambda market, symbol: bars[symbol]
        mock_drift.return_value = pd.DataFrame()

        run_accuracy_check(horizon=1)

        mock_save.assert_called_once()
        rows = {r["symbol"]: r for r in mock_save.call_args[0][0]}
        self.assertIn("GOOD", rows)
        self.assertEqual(rows["GOOD"]["actual_price"], 1020.0)
        for r in rows.values():
            self.assertFalse(pd.isna(r["actual_price"]))

    def test_split_adjust_scale_returns_none_for_non_finite(self):
        """基準の終値・現在値が有限でないとき、_split_adjust_scale は None を返すこと"""
        from src.prediction.prediction_pipeline import _split_adjust_scale

        self.assertIsNone(_split_adjust_scale(100.0, float("nan")))
        self.assertIsNone(_split_adjust_scale(float("nan"), 100.0))
        self.assertIsNone(_split_adjust_scale(100.0, float("inf")))


if __name__ == "__main__":
    unittest.main()
