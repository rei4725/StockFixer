"""Unit tests for backtest_optimize_pipeline.py"""

import json
import os
import tempfile
import unittest
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# _frange  （純粋関数）
# ──────────────────────────────────────────────────────────────────


class TestFrange(unittest.TestCase):
    """_frange の純粋関数テスト"""

    def test_basic_range_includes_stop(self):
        """start から stop まで step 刻みの値を生成すること（stop を含む）"""
        from src.services.backtest_optimize_pipeline import _frange

        result = _frange(0.0, 0.04, 0.01)
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[-1], 0.04)

    def test_single_value_when_step_exceeds_range(self):
        """step が range を超える場合は 1 要素のみ"""
        from src.services.backtest_optimize_pipeline import _frange

        result = _frange(0.0, 0.01, 0.05)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.0)

    def test_empty_when_start_exceeds_stop(self):
        """start > stop で空リスト"""
        from src.services.backtest_optimize_pipeline import _frange

        result = _frange(0.05, 0.01, 0.01)
        self.assertEqual(result, [])

    def test_values_are_rounded(self):
        """値が round(val, 6) で丸められていること"""
        from src.services.backtest_optimize_pipeline import _frange

        result = _frange(0.0, 0.03, 0.01)
        for v in result:
            self.assertEqual(v, round(v, 6))

    def test_three_step_range(self):
        """0.01 刻みで 3 値が生成されること"""
        from src.services.backtest_optimize_pipeline import _frange

        result = _frange(0.01, 0.02, 0.005)
        # [0.01, 0.015, 0.02] → 3 要素
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.01)
        self.assertAlmostEqual(result[-1], 0.02)


# ──────────────────────────────────────────────────────────────────
# _parse_grid_values  （純粋関数）
# ──────────────────────────────────────────────────────────────────


class TestParseGridValues(unittest.TestCase):
    """_parse_grid_values の純粋関数テスト"""

    def test_uses_provided_values(self):
        """指定した values がそのまま返ること"""
        from src.services.backtest_optimize_pipeline import _parse_grid_values

        result = _parse_grid_values([0.02, 0.03, 0.04], 0.01)
        self.assertEqual(result, [0.02, 0.03, 0.04])

    def test_uses_default_when_empty(self):
        """values が空のとき default が返ること"""
        from src.services.backtest_optimize_pipeline import _parse_grid_values

        result = _parse_grid_values([], 0.01)
        self.assertEqual(result, [0.01])

    def test_uses_default_when_none(self):
        """values が None のとき default が返ること"""
        from src.services.backtest_optimize_pipeline import _parse_grid_values

        result = _parse_grid_values(None, 0.02)
        self.assertEqual(result, [0.02])

    def test_values_are_converted_to_float(self):
        """整数値が float に変換されること"""
        from src.services.backtest_optimize_pipeline import _parse_grid_values

        result = _parse_grid_values([1, 2, 3], 0.0)
        self.assertTrue(all(isinstance(v, float) for v in result))


# ──────────────────────────────────────────────────────────────────
# get_optimal_params  （JSON 読み込み）
# ──────────────────────────────────────────────────────────────────


class TestGetOptimalParams(unittest.TestCase):
    """get_optimal_params の JSON 読み込みテスト"""

    def test_returns_none_when_file_missing(self):
        """ファイルが存在しない場合 None が返ること"""
        from src.services.backtest_optimize_pipeline import get_optimal_params

        result = get_optimal_params("jp", "7203", json_path="/nonexistent/params.json")
        self.assertIsNone(result)

    def test_returns_params_when_key_exists(self):
        """JSON ファイルにシンボルキーがある場合パラメータが返ること"""
        from src.services.backtest_optimize_pipeline import get_optimal_params

        params = {"threshold": 0.02, "fee_rate": 0.001}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"jp_7203": params}, f)
            tmp_file = f.name
        try:
            result = get_optimal_params("jp", "7203", json_path=tmp_file)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)
            self.assertAlmostEqual(result["threshold"], 0.02)
        finally:
            os.unlink(tmp_file)

    def test_returns_none_when_key_not_found(self):
        """JSON ファイルにシンボルキーがない場合 None が返ること"""
        from src.services.backtest_optimize_pipeline import get_optimal_params

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"us_AAPL": {"threshold": 0.01}}, f)
            tmp_file = f.name
        try:
            result = get_optimal_params("jp", "7203", json_path=tmp_file)
            self.assertIsNone(result)
        finally:
            os.unlink(tmp_file)


# ──────────────────────────────────────────────────────────────────
# save_optimization_results
# ──────────────────────────────────────────────────────────────────


class TestSaveOptimizationResults(unittest.TestCase):
    """save_optimization_results のテスト"""

    def test_saves_csv_file(self):
        """最適化結果が CSV として保存されること"""
        from src.services.backtest_optimize_pipeline import save_optimization_results

        result_df = pd.DataFrame(
            {
                "threshold": [0.01, 0.02, 0.03],
                "sharpe_ratio": [1.2, 1.5, 0.9],
                "total_return": [0.08, 0.12, 0.05],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.services.backtest_optimize_pipeline.get_results_dir",
                return_value=tmp_dir,
            ):
                path = save_optimization_results(result_df, "jp", "7203")

        self.assertIsNotNone(path)
        self.assertTrue(path.endswith(".csv"))

    def test_saved_csv_is_readable(self):
        """保存した CSV が読み込み可能であること"""
        from src.services.backtest_optimize_pipeline import save_optimization_results

        result_df = pd.DataFrame(
            {
                "threshold": [0.01, 0.02],
                "sharpe_ratio": [1.2, 1.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.services.backtest_optimize_pipeline.get_results_dir",
                return_value=tmp_dir,
            ):
                path = save_optimization_results(result_df, "jp", "7203")
            loaded = pd.read_csv(path)

        self.assertEqual(len(loaded), 2)
        self.assertIn("threshold", loaded.columns)


# ──────────────────────────────────────────────────────────────────
# run_optimization  （run_backtest_walk_forward をモック）
# ──────────────────────────────────────────────────────────────────


class TestRunOptimization(unittest.TestCase):
    """run_optimization のテスト（モック使用）"""

    def _make_wf_df(self):
        return pd.DataFrame(
            {
                "total_return": [0.12],
                "sharpe_ratio": [1.5],
                "max_drawdown": [-0.08],
                "win_rate": [0.6],
                "num_trades": [50],
                "gross_total_return": [0.13],
                "gross_sharpe_ratio": [1.6],
                "gross_max_drawdown": [-0.07],
                "cost_impact_return": [-0.01],
                "cost_impact_cash": [-1000.0],
                "profit_factor": [1.8],
                "avg_position_fraction": [0.5],
                "max_position_fraction": [1.0],
                "avg_position_value": [500000.0],
                "atr_fallback_trades": [0],
            }
        )

    @patch("src.services.backtest_optimize_pipeline.run_backtest_walk_forward")
    def test_returns_dataframe_with_results(self, mock_wf):
        """最適化結果が DataFrame として返ること"""
        from src.services.backtest_optimize_pipeline import run_optimization

        mock_wf.return_value = (None, None, self._make_wf_df())

        result = run_optimization(
            market="jp",
            symbol="7203",
            model_type="XGBoostModel",
            threshold_min=0.01,
            threshold_max=0.02,
            threshold_step=0.01,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreaterEqual(len(result), 1)
        self.assertIn("threshold", result.columns)

    @patch("src.services.backtest_optimize_pipeline.run_backtest_walk_forward")
    def test_returns_dataframe_on_all_errors(self, mock_wf):
        """全パラメータでエラーが発生しても DataFrame が返ること（エラー列を含む）"""
        from src.services.backtest_optimize_pipeline import run_optimization

        mock_wf.side_effect = Exception("バックテストエラー")

        result = run_optimization(
            market="jp",
            symbol="7203",
            model_type="XGBoostModel",
            threshold_min=0.01,
            threshold_max=0.01,
            threshold_step=0.01,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreaterEqual(len(result), 1)
        self.assertIn("error", result.columns)

    @patch("src.services.backtest_optimize_pipeline.run_backtest_walk_forward")
    def test_parameter_combinations_count(self, mock_wf):
        """閾値数 × ストップロス数 のパラメータ組み合わせが生成されること"""
        from src.services.backtest_optimize_pipeline import run_optimization

        mock_wf.return_value = (None, None, self._make_wf_df())

        # threshold: [0.01, 0.02] → 2パラメータ（optimize_risk=False → SL/TP各1）
        result = run_optimization(
            market="jp",
            symbol="7203",
            threshold_min=0.01,
            threshold_max=0.02,
            threshold_step=0.01,
        )
        self.assertEqual(len(result), 2)

    @patch("src.services.backtest_optimize_pipeline.run_backtest_walk_forward")
    def test_threshold_column_values_match_input(self, mock_wf):
        """結果の threshold 列が指定した閾値と一致すること"""
        from src.services.backtest_optimize_pipeline import run_optimization

        mock_wf.return_value = (None, None, self._make_wf_df())

        result = run_optimization(
            market="jp",
            symbol="7203",
            threshold_min=0.005,
            threshold_max=0.005,
            threshold_step=0.001,
        )
        self.assertAlmostEqual(float(result.iloc[0]["threshold"]), 0.005)


# ──────────────────────────────────────────────────────────────────
# save_optimal_params_json
# ──────────────────────────────────────────────────────────────────


class TestSaveOptimalParamsJson(unittest.TestCase):
    """save_optimal_params_json のテスト"""

    def _make_result_df(self):
        return pd.DataFrame(
            {
                "threshold": [0.02, 0.03],
                "sharpe_ratio": [1.5, 1.2],
                "total_return": [0.10, 0.08],
                "max_drawdown": [-0.05, -0.06],
                "win_rate": [0.6, 0.55],
                "num_trades": [50, 45],
                "gross_total_return": [0.11, 0.09],
                "gross_sharpe_ratio": [1.6, 1.3],
                "gross_max_drawdown": [-0.04, -0.05],
                "cost_impact_return": [-0.01, -0.01],
                "cost_impact_cash": [-1000.0, -800.0],
                "profit_factor": [1.8, 1.6],
                "avg_position_fraction": [0.5, 0.5],
                "max_position_fraction": [1.0, 1.0],
                "avg_position_value": [500000.0, 480000.0],
                "atr_fallback_trades": [0, 0],
            }
        )

    def test_saves_json_with_params(self):
        """最適パラメータが JSON ファイルに保存されること"""
        from unittest.mock import mock_open as mopen

        from src.services.backtest_optimize_pipeline import save_optimal_params_json

        m = mopen()
        with patch("os.path.exists", return_value=False), patch(
            "src.services.backtest_optimize_pipeline.ensure_dir"
        ), patch("builtins.open", m):
            path = save_optimal_params_json(
                self._make_result_df(), "jp", "7203", sort_by="sharpe_ratio"
            )

        self.assertIsNotNone(path)
        self.assertNotEqual(path, "")
        self.assertTrue(path.endswith("optimal_params.json"))

    def test_returns_empty_on_empty_df(self):
        """DataFrame が空の場合、空文字列が返ること"""
        from src.services.backtest_optimize_pipeline import save_optimal_params_json

        result = save_optimal_params_json(pd.DataFrame(), "jp", "7203")
        self.assertEqual(result, "")

    def test_merges_with_existing_json(self):
        """既存 JSON と結果が統合されること"""
        import json as _json

        from src.services.backtest_optimize_pipeline import save_optimal_params_json

        existing_data = {"jp_AAPL": {"threshold": 0.01}}
        captured: dict = {}

        def fake_dump(obj, fp, **kwargs):
            captured["obj"] = obj

        m = mock_open(read_data=_json.dumps(existing_data))
        with patch("os.path.exists", return_value=True), patch(
            "src.services.backtest_optimize_pipeline.ensure_dir"
        ), patch("builtins.open", m), patch("json.dump", side_effect=fake_dump):
            path = save_optimal_params_json(
                self._make_result_df(), "jp", "7203", sort_by="sharpe_ratio"
            )

        self.assertIsNotNone(path)
        self.assertNotEqual(path, "")
        if captured:
            merged = captured["obj"]
            self.assertIsInstance(merged, dict)
            self.assertIn("jp_AAPL", merged)
            self.assertIn("jp_7203", merged)


# ──────────────────────────────────────────────────────────────────
# run_optimize_batch
# ──────────────────────────────────────────────────────────────────


class TestRunOptimizeBatch(unittest.TestCase):
    """run_optimize_batch のテスト"""

    @patch("src.services.backtest_optimize_pipeline.save_optimal_params_json")
    @patch("src.services.backtest_optimize_pipeline.save_optimization_results")
    @patch("src.services.backtest_optimize_pipeline.run_optimization")
    @patch("src.services.batch_runner.load_target_symbols")
    def test_returns_list_of_results(self, mock_symbols, mock_run, mock_save, mock_json):
        """全銘柄の結果リストが返ること"""
        from src.domain.types import SymbolTask
        from src.services.backtest_optimize_pipeline import run_optimize_batch

        mock_symbols.return_value = [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="jp", symbol="9984"),
        ]
        mock_run.return_value = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        results = run_optimize_batch(model_type="XGBoostModel", max_workers=1)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    @patch("src.services.backtest_optimize_pipeline.save_optimal_params_json")
    @patch("src.services.backtest_optimize_pipeline.save_optimization_results")
    @patch("src.services.backtest_optimize_pipeline.run_optimization")
    @patch("src.services.batch_runner.load_target_symbols")
    def test_handles_single_symbol_error(self, mock_symbols, mock_run, mock_save, mock_json):
        """1銘柄でエラーが出ても他の銘柄が処理されること"""
        from src.domain.types import SymbolTask
        from src.services.backtest_optimize_pipeline import run_optimize_batch

        mock_symbols.return_value = [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="jp", symbol="9984"),
        ]
        mock_run.side_effect = [
            Exception("最適化エラー"),
            pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]}),
        ]
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        results = run_optimize_batch(model_type="XGBoostModel", max_workers=1)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        statuses = [r.get("status") for r in results]
        self.assertIn("error", statuses)

    @patch("src.services.batch_runner.load_target_symbols")
    def test_returns_empty_on_no_symbols(self, mock_symbols):
        """銘柄がない場合、空リストが返ること"""
        from src.services.backtest_optimize_pipeline import run_optimize_batch

        mock_symbols.return_value = []
        results = run_optimize_batch(model_type="XGBoostModel")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
