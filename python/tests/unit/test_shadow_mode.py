"""シャドーモード A/B テスト基盤（R-207）のユニットテスト"""

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.domain.types import PredictionResult
from src.utils.db.prediction import (
    load_shadow_comparison,
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


# ---------------------------------------------------------------------------
# PredictionResult の model_version フィールドテスト
# ---------------------------------------------------------------------------


class TestPredictionResultModelVersion(unittest.TestCase):
    """PredictionResult の model_version フィールドに関するテスト"""

    def test_default_model_version_is_none(self):
        """デフォルトで model_version は None"""
        r = PredictionResult("us", "AAPL", 100.0, 102.0, 0.02, 2)
        self.assertIsNone(r.model_version)

    def test_model_version_can_be_set(self):
        """model_version を任意の値に設定できること"""
        r = PredictionResult("us", "AAPL", 100.0, 102.0, 0.02, 2, model_version="challenger")
        self.assertEqual(r.model_version, "challenger")

    def test_to_dataframe_includes_model_version(self):
        """to_dataframe が model_version 列を含むこと"""
        results = [
            PredictionResult("us", "AAPL", 100.0, 102.0, 0.02, 2, model_version="production"),
            PredictionResult("us", "MSFT", 200.0, 206.0, 0.03, 2, model_version="challenger"),
        ]
        df = PredictionResult.to_dataframe(results)
        self.assertIn("model_version", df.columns)
        self.assertEqual(df.loc[df["symbol"] == "AAPL", "model_version"].iloc[0], "production")
        self.assertEqual(df.loc[df["symbol"] == "MSFT", "model_version"].iloc[0], "challenger")

    def test_to_dataframe_omits_model_version_when_none(self):
        """model_version が None の場合は列に含まれないこと"""
        results = [PredictionResult("us", "AAPL", 100.0, 102.0, 0.02, 2)]
        df = PredictionResult.to_dataframe(results)
        self.assertNotIn("model_version", df.columns)

    def test_from_dataframe_row_reads_model_version(self):
        """from_dataframe_row が model_version を正しく復元すること"""
        import pandas as pd

        row = pd.Series(
            {
                "market": "us",
                "symbol": "AAPL",
                "current_price": 100.0,
                "avg_pred_price": 102.0,
                "diff_ratio": 0.02,
                "model_count": 2,
                "model_version": "challenger",
            }
        )
        result = PredictionResult.from_dataframe_row(row)
        self.assertEqual(result.model_version, "challenger")

    def test_from_dataframe_row_handles_none_model_version(self):
        """from_dataframe_row が model_version=None を正しく扱うこと"""
        import pandas as pd

        row = pd.Series(
            {
                "market": "us",
                "symbol": "AAPL",
                "current_price": 100.0,
                "avg_pred_price": 102.0,
                "diff_ratio": 0.02,
                "model_count": 2,
            }
        )
        result = PredictionResult.from_dataframe_row(row)
        self.assertIsNone(result.model_version)


# ---------------------------------------------------------------------------
# save_prediction_results の model_version 対応テスト
# ---------------------------------------------------------------------------


class TestSavePredictionResultsWithModelVersion(_TmpDbTestCase):
    """save_prediction_results の model_version 対応テスト"""

    def test_production_and_challenger_coexist(self):
        """production / challenger 両バージョンが同一テーブルに共存できること"""
        prod = PredictionResult(
            "us", "AAPL", 100.0, 102.0, 0.02, 2, model_version="production"
        )
        chal = PredictionResult(
            "us", "AAPL", 100.0, 103.0, 0.03, 2, model_version="challenger"
        )
        save_prediction_results("20260403_120000", [prod])
        save_prediction_results("20260403_120000", [chal])

        from src.utils.db.prediction import load_prediction_results

        # model_version なしのクエリでは全バージョン取得（デフォルト動作は変更なし）
        # shadow_comparison で両バージョンを取得できること
        df = load_shadow_comparison(
            production_version="production",
            challenger_version="challenger",
        )
        self.assertEqual(len(df), 2)
        versions = set(df["model_version"].tolist())
        self.assertIn("production", versions)
        self.assertIn("challenger", versions)

    def test_delete_does_not_remove_other_version(self):
        """production を再保存しても challenger が削除されないこと"""
        prod = PredictionResult(
            "us", "AAPL", 100.0, 102.0, 0.02, 2, model_version="production"
        )
        chal = PredictionResult(
            "us", "AAPL", 100.0, 103.0, 0.03, 2, model_version="challenger"
        )
        save_prediction_results("20260403_120000", [prod])
        save_prediction_results("20260403_120000", [chal])

        # production を再度保存（上書き）
        prod2 = replace(prod, avg_pred_price=104.0)
        save_prediction_results("20260403_120000", [prod2])

        df = load_shadow_comparison(
            production_version="production",
            challenger_version="challenger",
        )
        self.assertEqual(len(df), 2)
        prod_row = df[df["model_version"] == "production"].iloc[0]
        self.assertAlmostEqual(prod_row["avg_pred_price"], 104.0, places=3)

    def test_default_model_version_is_production(self):
        """model_version が未設定の場合 'production' がデフォルト値として保存されること"""
        result = PredictionResult("us", "AAPL", 100.0, 102.0, 0.02, 2)
        save_prediction_results("20260403_120000", [result])

        df = load_shadow_comparison(
            production_version="production",
            challenger_version="challenger",
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(df["model_version"].iloc[0], "production")


# ---------------------------------------------------------------------------
# load_shadow_comparison テスト
# ---------------------------------------------------------------------------


class TestLoadShadowComparison(_TmpDbTestCase):
    """load_shadow_comparison のテスト"""

    def setUp(self):
        super().setUp()
        # production / challenger 両バージョンを保存
        prod_rows = [
            PredictionResult("us", "AAPL", 150.0, 153.0, 0.02, 2, model_version="production"),
            PredictionResult("us", "MSFT", 300.0, 309.0, 0.03, 2, model_version="production"),
            PredictionResult("jp", "7203", 3000.0, 3090.0, 0.03, 2, model_version="production"),
        ]
        chal_rows = [
            PredictionResult("us", "AAPL", 150.0, 154.5, 0.03, 2, model_version="challenger"),
            PredictionResult("us", "MSFT", 300.0, 312.0, 0.04, 2, model_version="challenger"),
        ]
        save_prediction_results("20260403_120000", prod_rows)
        save_prediction_results("20260403_120000", chal_rows)

    def test_returns_both_versions(self):
        """production / challenger 両バージョンが取得できること"""
        df = load_shadow_comparison()
        self.assertFalse(df.empty)
        versions = set(df["model_version"].tolist())
        self.assertIn("production", versions)
        self.assertIn("challenger", versions)

    def test_market_filter(self):
        """market フィルタが機能すること"""
        df = load_shadow_comparison(market="jp")
        self.assertTrue((df["market"] == "jp").all())

    def test_symbol_filter(self):
        """symbol フィルタが機能すること"""
        df = load_shadow_comparison(symbol="AAPL")
        self.assertTrue((df["symbol"] == "AAPL").all())

    def test_returns_empty_when_no_data(self):
        """データがない場合は空 DataFrame を返すこと"""
        df = load_shadow_comparison(
            production_version="v_nonexistent",
            challenger_version="v_nonexistent2",
        )
        self.assertTrue(df.empty)

    def test_version_count(self):
        """各バージョンの件数が正しいこと"""
        df = load_shadow_comparison(market="us")
        prod_count = (df["model_version"] == "production").sum()
        chal_count = (df["model_version"] == "challenger").sum()
        self.assertEqual(prod_count, 2)
        self.assertEqual(chal_count, 2)


# ---------------------------------------------------------------------------
# output_top_worst_results の shadow_mode テスト
# ---------------------------------------------------------------------------


class TestOutputTopWorstResultsShadowMode(unittest.TestCase):
    """output_top_worst_results の shadow_mode 引数に関するテスト"""

    def test_shadow_mode_tags_model_version(self):
        """shadow_mode=True のとき model_version が付与されて保存されること"""
        results = [PredictionResult("us", "AAPL", 100.0, 102.0, 0.02, 2)]
        saved_calls = []

        def mock_save(predicted_at, rows):
            saved_calls.append((predicted_at, rows))

        with patch(
            "src.services.prediction_pipeline.save_prediction_results",
            side_effect=mock_save,
        ):
            from src.services.prediction_pipeline import output_top_worst_results

            output_top_worst_results(
                results,
                shadow_mode=True,
                model_version="challenger",
            )

        self.assertEqual(len(saved_calls), 1)
        _ts, saved_rows = saved_calls[0]
        self.assertEqual(saved_rows[0].model_version, "challenger")

    def test_shadow_mode_false_does_not_override_version(self):
        """shadow_mode=False のとき model_version を上書きしないこと"""
        results = [
            PredictionResult("us", "AAPL", 100.0, 102.0, 0.02, 2, model_version="my_version")
        ]
        saved_calls = []

        def mock_save(predicted_at, rows):
            saved_calls.append((predicted_at, rows))

        with patch(
            "src.services.prediction_pipeline.save_prediction_results",
            side_effect=mock_save,
        ):
            from src.services.prediction_pipeline import output_top_worst_results

            output_top_worst_results(results, shadow_mode=False)

        _ts, saved_rows = saved_calls[0]
        # shadow_mode=False でも元のモデルバージョンはそのまま
        self.assertEqual(saved_rows[0].model_version, "my_version")

    def test_shadow_mode_preserves_existing_version(self):
        """shadow_mode=True でも既に model_version が設定済みの行は上書きしないこと"""
        results = [
            PredictionResult(
                "us", "AAPL", 100.0, 102.0, 0.02, 2, model_version="custom_version"
            )
        ]
        saved_calls = []

        def mock_save(predicted_at, rows):
            saved_calls.append((predicted_at, rows))

        with patch(
            "src.services.prediction_pipeline.save_prediction_results",
            side_effect=mock_save,
        ):
            from src.services.prediction_pipeline import output_top_worst_results

            output_top_worst_results(
                results,
                shadow_mode=True,
                model_version="challenger",
            )

        _ts, saved_rows = saved_calls[0]
        # 既存の model_version="custom_version" を保持する
        self.assertEqual(saved_rows[0].model_version, "custom_version")

    def test_no_results_returns_early(self):
        """結果が空の場合は保存されないこと"""
        saved_calls = []

        def mock_save(predicted_at, rows):
            saved_calls.append((predicted_at, rows))

        with patch(
            "src.services.prediction_pipeline.save_prediction_results",
            side_effect=mock_save,
        ):
            from src.services.prediction_pipeline import output_top_worst_results

            output_top_worst_results([], shadow_mode=True, model_version="challenger")

        self.assertEqual(len(saved_calls), 0)


# ---------------------------------------------------------------------------
# evaluate_shadow_models テスト
# ---------------------------------------------------------------------------


class TestEvaluateShadowModels(_TmpDbTestCase):
    """evaluate_shadow_models のテスト"""

    def _insert_accuracy_rows(self, model_name: str, rows: list[dict]):
        from src.utils.db.prediction import save_prediction_accuracy

        for row in rows:
            row.setdefault("model_name", model_name)
        save_prediction_accuracy(rows)

    def test_returns_dict_with_expected_keys(self):
        """戻り値が期待するキーを持つこと"""
        from src.services.shadow_evaluation_pipeline import evaluate_shadow_models

        # accuracy データなし
        result = evaluate_shadow_models(
            production_version="no_prod",
            challenger_version="no_chal",
        )
        expected_keys = {
            "production_hit_rate",
            "production_sharpe",
            "challenger_hit_rate",
            "challenger_sharpe",
            "challenger_wins",
            "evaluated_at",
            "n_production",
            "n_challenger",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_challenger_wins_false_when_no_data(self):
        """accuracy データがない場合は challenger_wins=False"""
        from src.services.shadow_evaluation_pipeline import evaluate_shadow_models

        result = evaluate_shadow_models()
        self.assertFalse(result["challenger_wins"])

    def test_challenger_wins_true_when_better(self):
        """challenger が production より Hit Rate / Sharpe ともに高い場合 True"""
        from src.services.shadow_evaluation_pipeline import evaluate_shadow_models

        # production: hit_rate=0.5
        prod_rows = [
            {
                "market": "us",
                "symbol": "AAPL",
                "model_name": "prod_v1",
                "predicted_at": f"2026040{i}_120000",
                "horizon": 1,
                "predicted_price": 102.0,
                "actual_price": 103.0 if i % 2 == 0 else 99.0,
                "predicted_ratio": 0.02,
                "actual_ratio": 0.03 if i % 2 == 0 else -0.01,
                "direction_match": i % 2 == 0,
            }
            for i in range(1, 9)
        ]
        # challenger: hit_rate=0.875
        chal_rows = [
            {
                "market": "us",
                "symbol": "AAPL",
                "model_name": "chal_v1",
                "predicted_at": f"2026040{i}_120000",
                "horizon": 1,
                "predicted_price": 102.0,
                "actual_price": 103.0 if i <= 7 else 99.0,
                "predicted_ratio": 0.02,
                "actual_ratio": 0.03 if i <= 7 else -0.01,
                "direction_match": i <= 7,
            }
            for i in range(1, 9)
        ]
        self._insert_accuracy_rows("prod_v1", prod_rows)
        self._insert_accuracy_rows("chal_v1", chal_rows)

        result = evaluate_shadow_models(
            production_version="prod_v1",
            challenger_version="chal_v1",
        )
        self.assertAlmostEqual(result["challenger_hit_rate"], 0.875, places=3)
        self.assertGreater(
            result["challenger_hit_rate"], result["production_hit_rate"]
        )
        self.assertTrue(result["challenger_wins"])

    def test_n_counts_match_inserted_rows(self):
        """n_production / n_challenger が挿入件数と一致すること"""
        from src.services.shadow_evaluation_pipeline import evaluate_shadow_models

        prod_rows = [
            {
                "market": "us",
                "symbol": "AAPL",
                "model_name": "prod",
                "predicted_at": f"2026040{i}_120000",
                "horizon": 1,
                "predicted_price": 102.0,
                "actual_price": 103.0,
                "predicted_ratio": 0.02,
                "actual_ratio": 0.03,
                "direction_match": True,
            }
            for i in range(1, 6)
        ]
        self._insert_accuracy_rows("prod", prod_rows)

        result = evaluate_shadow_models(
            production_version="prod",
            challenger_version="chal_nonexistent",
        )
        self.assertEqual(result["n_production"], 5)
        self.assertEqual(result["n_challenger"], 0)

    def test_records_to_experiment_runs(self):
        """評価結果が experiment_runs テーブルに記録されること"""
        from src.services.shadow_evaluation_pipeline import evaluate_shadow_models
        from src.utils.db.experiment import load_experiment_runs

        evaluate_shadow_models(
            production_version="prod_rec",
            challenger_version="chal_rec",
        )
        df = load_experiment_runs(model_name="prod_rec")
        self.assertFalse(df.empty)
        df_c = load_experiment_runs(model_name="chal_rec")
        self.assertFalse(df_c.empty)


# ---------------------------------------------------------------------------
# _compute_hit_rate / _compute_sharpe 単体テスト
# ---------------------------------------------------------------------------


class TestComputeMetrics(unittest.TestCase):
    """_compute_hit_rate / _compute_sharpe のユニットテスト"""

    def test_hit_rate_all_correct(self):
        import pandas as pd
        from src.services.shadow_evaluation_pipeline import _compute_hit_rate

        df = pd.DataFrame({"direction_match": [True, True, True, True]})
        self.assertAlmostEqual(_compute_hit_rate(df), 1.0)

    def test_hit_rate_all_wrong(self):
        import pandas as pd
        from src.services.shadow_evaluation_pipeline import _compute_hit_rate

        df = pd.DataFrame({"direction_match": [False, False, False]})
        self.assertAlmostEqual(_compute_hit_rate(df), 0.0)

    def test_hit_rate_returns_none_on_empty(self):
        import pandas as pd
        from src.services.shadow_evaluation_pipeline import _compute_hit_rate

        self.assertIsNone(_compute_hit_rate(pd.DataFrame()))

    def test_sharpe_positive_when_all_correct(self):
        import pandas as pd
        from src.services.shadow_evaluation_pipeline import _compute_sharpe

        df = pd.DataFrame(
            {
                "predicted_ratio": [0.02, 0.03, 0.01, 0.02],
                "actual_ratio": [0.03, 0.04, 0.02, 0.03],
            }
        )
        sharpe = _compute_sharpe(df)
        self.assertIsNotNone(sharpe)
        self.assertGreater(sharpe, 0)

    def test_sharpe_returns_none_on_insufficient_data(self):
        import pandas as pd
        from src.services.shadow_evaluation_pipeline import _compute_sharpe

        df = pd.DataFrame(
            {"predicted_ratio": [0.02], "actual_ratio": [0.03]}
        )
        self.assertIsNone(_compute_sharpe(df))

    def test_sharpe_returns_none_on_empty(self):
        import pandas as pd
        from src.services.shadow_evaluation_pipeline import _compute_sharpe

        self.assertIsNone(_compute_sharpe(pd.DataFrame()))


# ---------------------------------------------------------------------------
# run_shadow_prediction テスト
# ---------------------------------------------------------------------------


class TestRunShadowPrediction(unittest.TestCase):
    """run_shadow_prediction のテスト"""

    def test_calls_both_predict_fns(self):
        """production / challenger の両予測関数が呼ばれること"""
        from unittest.mock import MagicMock, patch

        prod_result = [PredictionResult("us", "AAPL", 100.0, 102.0, 0.02, 2)]
        chal_result = [PredictionResult("us", "AAPL", 100.0, 103.0, 0.03, 2)]

        prod_fn = MagicMock(return_value=prod_result)
        chal_fn = MagicMock(return_value=chal_result)

        with patch(
            "src.services.prediction_pipeline.save_prediction_results",
        ):
            from src.services.shadow_evaluation_pipeline import run_shadow_prediction

            result = run_shadow_prediction(
                predict_fn=prod_fn,
                challenger_predict_fn=chal_fn,
            )

        prod_fn.assert_called_once()
        chal_fn.assert_called_once()
        self.assertEqual(result["production_count"], 1)
        self.assertEqual(result["challenger_count"], 1)

    def test_uses_same_fn_when_no_challenger_fn(self):
        """challenger_predict_fn が None のとき predict_fn を2回呼ぶこと"""
        from unittest.mock import MagicMock, patch

        result_data = [PredictionResult("us", "AAPL", 100.0, 102.0, 0.02, 2)]
        fn = MagicMock(return_value=result_data)

        with patch(
            "src.services.prediction_pipeline.save_prediction_results",
        ):
            from src.services.shadow_evaluation_pipeline import run_shadow_prediction

            run_shadow_prediction(predict_fn=fn, challenger_predict_fn=None)

        self.assertEqual(fn.call_count, 2)
