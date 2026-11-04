"""Unit tests for backtest_optimize_pipeline.py"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

# ──────────────────────────────────────────────────────────────────
# _frange  （純粋関数）
# ──────────────────────────────────────────────────────────────────


class TestFrange(unittest.TestCase):
    """_frange の純粋関数テスト"""

    def test_basic_range_includes_stop(self):
        """start から stop まで step 刻みの値を生成すること（stop を含む）"""
        from src.backtest.optimizer import _frange

        result = _frange(0.0, 0.04, 0.01)
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[-1], 0.04)

    def test_single_value_when_step_exceeds_range(self):
        """step が range を超える場合は 1 要素のみ"""
        from src.backtest.optimizer import _frange

        result = _frange(0.0, 0.01, 0.05)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.0)

    def test_empty_when_start_exceeds_stop(self):
        """start > stop で空リスト"""
        from src.backtest.optimizer import _frange

        result = _frange(0.05, 0.01, 0.01)
        self.assertEqual(result, [])

    def test_values_are_rounded(self):
        """値が round(val, 6) で丸められていること"""
        from src.backtest.optimizer import _frange

        result = _frange(0.0, 0.03, 0.01)
        for v in result:
            self.assertEqual(v, round(v, 6))

    def test_three_step_range(self):
        """0.01 刻みで 3 値が生成されること"""
        from src.backtest.optimizer import _frange

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
        from src.backtest.optimizer import _parse_grid_values

        result = _parse_grid_values([0.02, 0.03, 0.04], 0.01)
        self.assertEqual(result, [0.02, 0.03, 0.04])

    def test_uses_default_when_empty(self):
        """values が空のとき default が返ること"""
        from src.backtest.optimizer import _parse_grid_values

        result = _parse_grid_values([], 0.01)
        self.assertEqual(result, [0.01])

    def test_uses_default_when_none(self):
        """values が None のとき default が返ること"""
        from src.backtest.optimizer import _parse_grid_values

        result = _parse_grid_values(None, 0.02)
        self.assertEqual(result, [0.02])

    def test_values_are_converted_to_float(self):
        """整数値が float に変換されること"""
        from src.backtest.optimizer import _parse_grid_values

        result = _parse_grid_values([1, 2, 3], 0.0)
        self.assertTrue(all(isinstance(v, float) for v in result))


# ──────────────────────────────────────────────────────────────────
# get_optimal_params  （JSON 読み込み）
# ──────────────────────────────────────────────────────────────────


class TestGetOptimalParams(unittest.TestCase):
    """get_optimal_params の JSON 読み込みテスト"""

    def test_returns_none_when_file_missing(self):
        """ファイルが存在しない場合 None が返ること"""
        from src.backtest.optimizer import get_optimal_params

        result = get_optimal_params("jp", "7203", json_path="/nonexistent/params.json")
        self.assertIsNone(result)

    def test_returns_params_when_key_exists(self):
        """JSON ファイルにシンボルキーがある場合パラメータが返ること"""
        from src.backtest.optimizer import get_optimal_params

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
        from src.backtest.optimizer import get_optimal_params

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

    def test_raises_when_json_is_broken(self):
        """ファイルが在るのに JSON が壊れている場合は None ではなく例外を投げること（#931）"""
        from src.backtest.optimizer import get_optimal_params
        from src.backtest.optimizer.persistence import OptimalParamsCorruptedError

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, "optimal_params.json")
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write('{"jp_7203": {"threshold": 0.0')
            with self.assertRaises(OptimalParamsCorruptedError):
                get_optimal_params("jp", "7203", json_path=tmp_file)
            # 読み手はファイルを動かさない
            self.assertTrue(os.path.exists(tmp_file))

    def test_raises_when_top_level_is_list(self):
        """最上位が list の場合は None ではなく例外を投げること（#931）"""
        from src.backtest.optimizer import get_optimal_params
        from src.backtest.optimizer.persistence import OptimalParamsCorruptedError

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, "optimal_params.json")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(["jp_7203"], f)
            with self.assertRaises(OptimalParamsCorruptedError):
                get_optimal_params("jp", "7203", json_path=tmp_file)


# ──────────────────────────────────────────────────────────────────
# save_optimization_results
# ──────────────────────────────────────────────────────────────────


class TestSaveOptimizationResults(unittest.TestCase):
    """save_optimization_results のテスト"""

    def test_saves_csv_file(self):
        """最適化結果が CSV として保存されること"""
        from src.backtest.optimizer import save_optimization_results

        result_df = pd.DataFrame(
            {
                "threshold": [0.01, 0.02, 0.03],
                "sharpe_ratio": [1.2, 1.5, 0.9],
                "total_return": [0.08, 0.12, 0.05],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.utils.results_io.get_results_dir",
                return_value=tmp_dir,
            ):
                path = save_optimization_results(result_df, "jp", "7203")

        self.assertIsNotNone(path)
        self.assertTrue(path.endswith(".csv"))

    def test_saved_csv_is_readable(self):
        """保存した CSV が読み込み可能であること"""
        from src.backtest.optimizer import save_optimization_results

        result_df = pd.DataFrame(
            {
                "threshold": [0.01, 0.02],
                "sharpe_ratio": [1.2, 1.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.utils.results_io.get_results_dir",
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

    @patch("src.backtest.optimizer.grid.run_backtest_walk_forward")
    def test_returns_dataframe_with_results(self, mock_wf):
        """最適化結果が DataFrame として返ること"""
        from src.backtest.optimizer import run_optimization

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

    @patch("src.backtest.optimizer.grid.run_backtest_walk_forward")
    def test_returns_dataframe_on_all_errors(self, mock_wf):
        """全パラメータでエラーが発生しても DataFrame が返ること（エラー列を含む）"""
        from src.backtest.optimizer import run_optimization

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

    @patch("src.backtest.optimizer.grid.run_backtest_walk_forward")
    def test_parameter_combinations_count(self, mock_wf):
        """閾値数 × ストップロス数 のパラメータ組み合わせが生成されること"""
        from src.backtest.optimizer import run_optimization

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

    @patch("src.backtest.optimizer.grid.run_backtest_walk_forward")
    def test_threshold_column_values_match_input(self, mock_wf):
        """結果の threshold 列が指定した閾値と一致すること"""
        from src.backtest.optimizer import run_optimization

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


def _patch_config_root(root):
    """最適パラメータの保存先を root/config へ向ける（実ファイルに触れない）。"""
    from tests.unit.optimal_params_isolation import redirect_optimal_params

    return redirect_optimal_params(root)


def _save_and_read(df):
    """save_optimal_params_json を一時フォルダに実行し、書き込まれた dict を返す。"""
    from src.backtest.optimizer import save_optimal_params_json

    with tempfile.TemporaryDirectory() as root, _patch_config_root(root):
        path = save_optimal_params_json(df, "jp", "7203")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


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
        from src.backtest.optimizer import save_optimal_params_json

        with tempfile.TemporaryDirectory() as root, _patch_config_root(root):
            path = save_optimal_params_json(
                self._make_result_df(), "jp", "7203", sort_by="sharpe_ratio"
            )
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)

        self.assertIsNotNone(path)
        self.assertNotEqual(path, "")
        self.assertTrue(path.endswith("optimal_params.json"))
        self.assertIn("jp_7203", saved)

    def test_returns_empty_on_empty_df(self):
        """DataFrame が空の場合、空文字列が返ること"""
        from src.backtest.optimizer import save_optimal_params_json

        result = save_optimal_params_json(pd.DataFrame(), "jp", "7203")
        self.assertEqual(result, "")

    def test_merges_with_existing_json(self):
        """既存 JSON と結果が統合されること"""
        from src.backtest.optimizer import save_optimal_params_json

        existing_data = {"jp_AAPL": {"threshold": 0.01}}

        with tempfile.TemporaryDirectory() as root, _patch_config_root(root):
            os.makedirs(os.path.join(root, "config"))
            with open(os.path.join(root, "config", "optimal_params.json"), "w") as f:
                json.dump(existing_data, f)
            path = save_optimal_params_json(
                self._make_result_df(), "jp", "7203", sort_by="sharpe_ratio"
            )
            with open(path, "r", encoding="utf-8") as f:
                merged = json.load(f)

        self.assertIsNotNone(path)
        self.assertNotEqual(path, "")
        self.assertIsInstance(merged, dict)
        self.assertIn("jp_AAPL", merged)
        self.assertIn("jp_7203", merged)


# ──────────────────────────────────────────────────────────────────
# run_optimize_batch
# ──────────────────────────────────────────────────────────────────


class TestRunOptimizeBatch(unittest.TestCase):
    """run_optimize_batch のテスト"""

    def setUp(self):
        # 計算の前の特徴量の確かめは、既定では全銘柄に行があるものとする
        patcher = patch(
            "src.backtest.optimizer.batch.load_stock_features",
            return_value=pd.DataFrame({"y": [0.01]}),
        )
        self.mock_features = patcher.start()
        self.addCleanup(patcher.stop)

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    def test_no_features_skips_without_grid(self, mock_run, mock_save, mock_json):
        """特徴量の無い銘柄はグリッドを回さず skipped で、ある銘柄は計算と保存が変わらないこと（#1579）"""
        from src.backtest.optimizer import run_optimize_batch
        from src.domain.types import SymbolTask

        good_df = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        self.mock_features.side_effect = lambda market, symbol, **kwargs: (
            None if symbol == "VYLR" else pd.DataFrame({"y": [0.01]})
        )
        mock_run.return_value = good_df
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        with self.assertLogs("src.backtest.optimizer.batch", level="WARNING") as logs:
            results = run_optimize_batch(
                [SymbolTask(market="us", symbol="VYLR"), SymbolTask(market="us", symbol="AAPL")],
                model_type="XGBoostModel",
                max_workers=1,
            )

        by_symbol = {r["symbol"]: r for r in results}
        self.assertEqual(by_symbol["VYLR"], {"market": "us", "symbol": "VYLR", "status": "skipped"})
        self.assertEqual(by_symbol["AAPL"], {"market": "us", "symbol": "AAPL", "status": "success"})
        self.assertEqual([c.kwargs["symbol"] for c in mock_run.call_args_list], ["AAPL"])
        mock_save.assert_called_once_with(good_df, "us", "AAPL")
        mock_json.assert_called_once_with(good_df, "us", "AAPL", sort_by="sharpe_ratio")
        self.assertEqual(self.mock_features.call_count, 2)
        vylr_logs = [r for r in logs.records if "us/VYLR" in r.getMessage()]
        self.assertEqual([r.levelname for r in vylr_logs], ["WARNING"])

    @patch("src.backtest.optimizer.batch.run_optimization")
    def test_feature_check_failure_counts_as_error(self, mock_run):
        """特徴量の確かめが例外で落ちる日は skipped にせず error に数えること（#1579）"""
        from src.backtest.optimizer import run_optimize_batch
        from src.domain.types import SymbolTask

        self.mock_features.side_effect = RuntimeError("接続できません")

        results = run_optimize_batch(
            [SymbolTask(market="us", symbol="AAPL")],
            model_type="XGBoostModel",
            max_workers=1,
        )

        self.assertEqual(
            results,
            [{"market": "us", "symbol": "AAPL", "status": "error", "error": "接続できません"}],
        )
        self.assertEqual(self.mock_features.call_args.kwargs, {"raise_on_error": True})
        mock_run.assert_not_called()

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    def test_other_source_does_not_check_features(self, mock_run, mock_save, mock_json):
        """stock_features を読まない source では確かめずに今までどおり計算すること（#1579）"""
        from src.backtest.optimizer import run_optimize_batch
        from src.domain.types import SymbolTask

        mock_run.return_value = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        results = run_optimize_batch(
            [SymbolTask(market="us", symbol="AAPL")],
            model_type="XGBoostModel",
            source="api",
            max_workers=1,
        )

        self.assertEqual(results[0]["status"], "success")
        self.mock_features.assert_not_called()

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    def test_returns_list_of_results(self, mock_run, mock_save, mock_json):
        """全銘柄の結果リストが返ること"""
        from src.backtest.optimizer import run_optimize_batch
        from src.domain.types import SymbolTask

        tasks = [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="jp", symbol="9984"),
        ]
        mock_run.return_value = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        results = run_optimize_batch(tasks, model_type="XGBoostModel", max_workers=1)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    def test_handles_single_symbol_error(self, mock_run, mock_save, mock_json):
        """1銘柄でエラーが出ても他の銘柄が処理されること"""
        from src.backtest.optimizer import run_optimize_batch
        from src.domain.types import SymbolTask

        tasks = [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="jp", symbol="9984"),
        ]
        mock_run.side_effect = [
            Exception("最適化エラー"),
            pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]}),
        ]
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        results = run_optimize_batch(tasks, model_type="XGBoostModel", max_workers=1)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        statuses = [r.get("status") for r in results]
        self.assertIn("error", statuses)

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    @patch("src.backtest.optimizer.batch.get_optimal_params")
    def test_unreadable_existing_params_skip_overwrite(
        self, mock_get, mock_run, mock_save, mock_json
    ):
        """既存の結果が読めないときは無いとみなさず、最適化も上書きもしないこと（#931）"""
        from src.backtest.optimizer import run_optimize_batch
        from src.backtest.optimizer.persistence import OptimalParamsCorruptedError
        from src.domain.types import SymbolTask

        mock_get.side_effect = OptimalParamsCorruptedError("壊れた最適パラメータJSON")

        results = run_optimize_batch(
            [SymbolTask(market="jp", symbol="7203")],
            model_type="XGBoostModel",
            max_workers=1,
            skip_days=7,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "error")
        mock_run.assert_not_called()
        mock_save.assert_not_called()
        mock_json.assert_not_called()

    def test_returns_empty_on_no_tasks(self):
        """タスクが空の場合、空リストが返ること"""
        from src.backtest.optimizer import run_optimize_batch

        results = run_optimize_batch([], model_type="XGBoostModel")
        self.assertEqual(results, [])

    @patch("src.utils.db.load_stock_features")
    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    def test_feature_load_failure_keeps_other_symbols(
        self, mock_run, mock_save, mock_json, mock_load
    ):
        """1銘柄の特徴量が読めなくても、ほかの銘柄は保存され、その銘柄は error になること（#972）"""
        from src.backtest.optimizer import run_optimize_batch
        from src.backtest.pipeline import load_features
        from src.domain.types import SymbolTask

        mock_load.return_value = pd.DataFrame()
        good_df = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})

        def _run(market, symbol, **kwargs):
            if symbol == "7203":
                load_features(market, symbol, "file")
            return good_df

        mock_run.side_effect = _run
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        results = run_optimize_batch(
            [SymbolTask(market="jp", symbol="7203"), SymbolTask(market="jp", symbol="9984")],
            model_type="XGBoostModel",
            max_workers=1,
        )

        by_symbol = {r["symbol"]: r for r in results}
        self.assertEqual(by_symbol["7203"]["status"], "error")
        self.assertEqual(by_symbol["9984"]["status"], "success")
        mock_json.assert_called_once()
        self.assertEqual(mock_json.call_args.args[2], "9984")

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    def test_interrupt_keeps_already_computed_results(self, mock_run, mock_save, mock_json):
        """途中の銘柄で中断されても、それより前に計算が終わった銘柄は保存済みであること（#1094）"""
        import threading

        from src.backtest.optimizer import run_optimize_batch
        from src.domain.types import SymbolTask

        class _Interrupted(BaseException):
            pass

        good_df = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        saved_two = threading.Event()

        def _json(df, market, symbol, sort_by="sharpe_ratio"):
            if mock_json.call_count >= 2:
                saved_two.set()
            return "/tmp/params.json"

        def _run(market, symbol, **kwargs):
            if symbol == "6758":
                # 前の2銘柄の保存を待ってから中断する（待ち切れなければそのまま中断）
                saved_two.wait(timeout=2)
                raise _Interrupted()
            return good_df

        mock_run.side_effect = _run
        mock_save.return_value = "/tmp/results.csv"
        mock_json.side_effect = _json

        tasks = [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="jp", symbol="9984"),
            SymbolTask(market="jp", symbol="6758"),
        ]
        with self.assertRaises(_Interrupted):
            run_optimize_batch(tasks, model_type="XGBoostModel", max_workers=1)

        saved_symbols = sorted(c.args[2] for c in mock_json.call_args_list)
        self.assertEqual(saved_symbols, ["7203", "9984"])
        self.assertEqual(sorted(c.args[2] for c in mock_save.call_args_list), ["7203", "9984"])

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    @patch("src.backtest.optimizer.batch.get_optimal_params")
    def test_full_run_saves_same_contents(self, mock_get, mock_run, mock_save, mock_json):
        """全部終わったときの保存の呼び出しと結果サマリーが今までと同じであること（#1094）"""
        from datetime import datetime

        from src.backtest.optimizer import run_optimize_batch
        from src.domain.types import SymbolTask

        df_a = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        mock_get.side_effect = lambda m, s: (
            {"timestamp": datetime.now().isoformat()} if s == "6758" else None
        )

        def _run(market, symbol, **kwargs):
            if symbol == "9984":
                raise ValueError("最適化エラー")
            return df_a

        mock_run.side_effect = _run
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        results = run_optimize_batch(
            [
                SymbolTask(market="jp", symbol="7203"),
                SymbolTask(market="jp", symbol="9984"),
                SymbolTask(market="jp", symbol="6758"),
            ],
            model_type="XGBoostModel",
            max_workers=2,
            sort_by="total_return",
            skip_days=7,
        )

        by_symbol = {r["symbol"]: r for r in results}
        self.assertEqual(len(results), 3)
        self.assertEqual(by_symbol["7203"], {"market": "jp", "symbol": "7203", "status": "success"})
        self.assertEqual(
            by_symbol["9984"],
            {"market": "jp", "symbol": "9984", "status": "error", "error": "最適化エラー"},
        )
        self.assertEqual(by_symbol["6758"], {"market": "jp", "symbol": "6758", "status": "skipped"})
        mock_save.assert_called_once_with(df_a, "jp", "7203")
        mock_json.assert_called_once_with(df_a, "jp", "7203", sort_by="total_return")

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    def test_no_valid_rows_counts_as_error(self, mock_run, mock_save, mock_json):
        """全候補が error の行だけの銘柄は保存されず error になり、成功に数えないこと（#1285）"""
        from src.backtest.optimizer import run_optimize_batch
        from src.backtest.optimizer.persistence import save_optimal_params_json as real_save_json
        from src.domain.types import SymbolTask

        error_df = pd.DataFrame(
            {"threshold": [0.01, 0.02], "error": ["データが取得できませんでした"] * 2}
        )
        good_df = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})

        def _json(df, market, symbol, sort_by="sharpe_ratio"):
            if symbol == "7203":
                # 有効な行が無い表では、本物は何も書かずに "" を返す
                return real_save_json(df, market, symbol, sort_by=sort_by)
            return "/tmp/params.json"

        mock_run.side_effect = lambda market, symbol, **kwargs: (
            error_df if symbol == "7203" else good_df
        )
        mock_save.return_value = "/tmp/results.csv"
        mock_json.side_effect = _json

        results = run_optimize_batch(
            [SymbolTask(market="jp", symbol="7203"), SymbolTask(market="jp", symbol="9984")],
            model_type="XGBoostModel",
            max_workers=1,
        )

        by_symbol = {r["symbol"]: r for r in results}
        self.assertEqual(
            by_symbol["7203"],
            {"market": "jp", "symbol": "7203", "status": "error", "error": "有効な最適化結果なし"},
        )
        self.assertEqual(by_symbol["9984"], {"market": "jp", "symbol": "9984", "status": "success"})

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    @patch("src.backtest.optimizer.batch.get_optimal_params")
    def test_computes_oldest_optimized_first_and_jp_first_on_same_day(
        self, mock_get, mock_run, mock_save, mock_json
    ):
        """最後に最適化した日が古い銘柄から計算し、同じ日なら jp を先にすること（#1522）"""
        from src.backtest.optimizer import run_optimize_batch
        from src.domain.types import SymbolTask

        timestamps = {
            "AAPL": "2026-11-04T01:06:00",
            "MSFT": "2026-03-23T19:22:30",
            "GOOG": "2026-03-23T19:22:31",
            "7203": "2026-03-23T19:22:37",
            "9984": "2026-03-23T08:00:00",
            "BROKEN": "壊れた日時",
        }
        mock_get.side_effect = lambda market, symbol: (
            {"timestamp": timestamps[symbol]} if symbol in timestamps else None
        )
        mock_run.return_value = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        run_optimize_batch(
            [
                SymbolTask(market="us", symbol="AAPL"),
                SymbolTask(market="us", symbol="MSFT"),
                SymbolTask(market="us", symbol="GOOG"),
                SymbolTask(market="jp", symbol="7203"),
                SymbolTask(market="jp", symbol="9984"),
                SymbolTask(market="us", symbol="BROKEN"),
                {"market": "us", "symbol": "NEVER"},
            ],
            model_type="XGBoostModel",
            max_workers=1,
        )

        computed = [c.kwargs["symbol"] for c in mock_run.call_args_list]
        self.assertEqual(computed, ["BROKEN", "NEVER", "7203", "9984", "MSFT", "GOOG", "AAPL"])

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    @patch("src.backtest.optimizer.batch.get_optimal_params")
    def test_next_week_starts_from_symbols_not_reached(
        self, mock_get, mock_run, mock_save, mock_json
    ):
        """前の週に届かなかった銘柄から始まり、続きの銘柄が先頭の銘柄より先に計算されること（#1522）"""
        from datetime import datetime, timedelta

        from src.backtest.optimizer import run_optimize_batch
        from src.domain.types import SymbolTask

        # 先頭の2銘柄だけ前の週に保存された。7日たっているのでスキップされない
        last_week = (datetime.now() - timedelta(days=7)).isoformat()
        timestamps = {
            "AAPL": last_week,
            "AMZN": last_week,
            "MSFT": "2026-03-23T19:22:30",
        }
        mock_get.side_effect = lambda market, symbol: {"timestamp": timestamps[symbol]}
        mock_run.return_value = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        run_optimize_batch(
            [SymbolTask(market="us", symbol=symbol) for symbol in timestamps],
            model_type="XGBoostModel",
            max_workers=1,
            skip_days=6,
        )

        computed = [c.kwargs["symbol"] for c in mock_run.call_args_list]
        self.assertEqual(computed[0], "MSFT")

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.save_optimization_results")
    @patch("src.backtest.optimizer.batch.run_optimization")
    @patch("src.backtest.optimizer.batch.get_optimal_params")
    def test_unreadable_params_keep_given_order(self, mock_get, mock_run, mock_save, mock_json):
        """既存の結果が読めないときは渡された順のまま進み、読むのは1回で止めること（#1522）"""
        from src.backtest.optimizer import run_optimize_batch
        from src.backtest.optimizer.persistence import OptimalParamsCorruptedError
        from src.domain.types import SymbolTask

        mock_get.side_effect = OptimalParamsCorruptedError("壊れた最適パラメータJSON")
        mock_run.return_value = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        mock_save.return_value = "/tmp/results.csv"
        mock_json.return_value = "/tmp/params.json"

        run_optimize_batch(
            [SymbolTask(market="us", symbol="AAPL"), SymbolTask(market="jp", symbol="7203")],
            model_type="XGBoostModel",
            max_workers=1,
        )

        computed = [c.kwargs["symbol"] for c in mock_run.call_args_list]
        self.assertEqual(computed, ["AAPL", "7203"])
        mock_get.assert_called_once()


class TestRunOptunaBatch(unittest.TestCase):
    """run_optuna_batch のテスト"""

    @patch("src.utils.db.load_stock_features")
    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.run_optuna_optimization")
    def test_feature_load_failure_keeps_other_symbols(self, mock_run, mock_json, mock_load):
        """1銘柄の特徴量が読めなくても、ほかの銘柄は保存され、その銘柄は error になること（#972）"""
        from src.backtest.optimizer import run_optuna_batch
        from src.backtest.pipeline import load_features
        from src.domain.types import SymbolTask

        mock_load.return_value = pd.DataFrame()
        good_df = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})

        def _run(market, symbol, **kwargs):
            if symbol == "7203":
                load_features(market, symbol, "file")
            return good_df

        mock_run.side_effect = _run
        mock_json.return_value = "/tmp/params.json"

        results = run_optuna_batch(
            [SymbolTask(market="jp", symbol="7203"), SymbolTask(market="jp", symbol="9984")],
            n_trials=1,
            max_workers=2,
        )

        by_symbol = {r["symbol"]: r for r in results}
        self.assertEqual(by_symbol["7203"]["status"], "error")
        self.assertEqual(by_symbol["9984"]["status"], "success")
        mock_json.assert_called_once()
        self.assertEqual(mock_json.call_args.args[2], "9984")

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.run_optuna_optimization")
    def test_saves_finished_symbol_while_others_running(self, mock_run, mock_json):
        """ほかの銘柄が計算中でも、終わった銘柄の保存が呼ばれること（#1107）"""
        import threading

        from src.backtest.optimizer import run_optuna_batch
        from src.domain.types import SymbolTask

        good_df = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        saved_first = threading.Event()
        saved_while_running = []

        def _json(df, market, symbol, sort_by="sharpe_ratio"):
            if symbol == "7203":
                saved_first.set()
            return "/tmp/params.json"

        def _run(market, symbol, **kwargs):
            if symbol == "9984":
                # 計算中のまま、先に終わった 7203 の保存を待つ
                saved_while_running.append(saved_first.wait(timeout=2))
            return good_df

        mock_run.side_effect = _run
        mock_json.side_effect = _json

        results = run_optuna_batch(
            [SymbolTask(market="jp", symbol="7203"), SymbolTask(market="jp", symbol="9984")],
            n_trials=1,
            max_workers=2,
        )

        self.assertEqual(saved_while_running, [True])
        self.assertEqual(sorted(r["status"] for r in results), ["success", "success"])

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.run_optuna_optimization")
    def test_interrupt_keeps_already_computed_results(self, mock_run, mock_json):
        """途中の銘柄で中断されても、それより前に計算が終わった銘柄は保存済みであること（#1107）"""
        import threading

        from src.backtest.optimizer import run_optuna_batch
        from src.domain.types import SymbolTask

        class _Interrupted(BaseException):
            pass

        good_df = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})
        saved_two = threading.Event()

        def _json(df, market, symbol, sort_by="sharpe_ratio"):
            if mock_json.call_count >= 2:
                saved_two.set()
            return "/tmp/params.json"

        def _run(market, symbol, **kwargs):
            if symbol == "6758":
                # 前の2銘柄の保存を待ってから中断する（待ち切れなければそのまま中断）
                saved_two.wait(timeout=2)
                raise _Interrupted()
            return good_df

        mock_run.side_effect = _run
        mock_json.side_effect = _json

        tasks = [
            SymbolTask(market="jp", symbol="7203"),
            SymbolTask(market="jp", symbol="9984"),
            SymbolTask(market="jp", symbol="6758"),
        ]
        with self.assertRaises(_Interrupted):
            run_optuna_batch(tasks, n_trials=1, max_workers=1)

        saved_symbols = sorted(c.args[2] for c in mock_json.call_args_list)
        self.assertEqual(saved_symbols, ["7203", "9984"])

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.run_optuna_optimization")
    def test_full_run_summary_unchanged(self, mock_run, mock_json):
        """全部終わったときの保存の呼び出しと結果サマリーが今までと同じであること（#1107）"""
        from src.backtest.optimizer import run_optuna_batch
        from src.domain.types import SymbolTask

        df_a = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})

        def _run(market, symbol, **kwargs):
            if symbol == "9984":
                raise ValueError("最適化エラー")
            return df_a

        def _json(df, market, symbol, sort_by="sharpe_ratio"):
            if symbol == "6758":
                raise OSError("書き込みエラー")
            return "/tmp/params.json"

        mock_run.side_effect = _run
        mock_json.side_effect = _json

        results = run_optuna_batch(
            [
                SymbolTask(market="jp", symbol="7203"),
                SymbolTask(market="jp", symbol="9984"),
                SymbolTask(market="jp", symbol="6758"),
            ],
            n_trials=1,
            max_workers=2,
            sort_by="total_return",
        )

        by_symbol = {r["symbol"]: r for r in results}
        self.assertEqual(len(results), 3)
        self.assertEqual(by_symbol["7203"], {"market": "jp", "symbol": "7203", "status": "success"})
        self.assertEqual(
            by_symbol["9984"],
            {"market": "jp", "symbol": "9984", "status": "error", "error": "最適化エラー"},
        )
        self.assertEqual(
            by_symbol["6758"],
            {"market": "jp", "symbol": "6758", "status": "error", "error": "書き込みエラー"},
        )
        self.assertEqual(mock_json.call_count, 2)
        mock_json.assert_any_call(df_a, "jp", "7203", sort_by="total_return")

    @patch("src.backtest.optimizer.batch.save_optimal_params_json")
    @patch("src.backtest.optimizer.batch.run_optuna_optimization")
    def test_no_valid_rows_counts_as_error(self, mock_run, mock_json):
        """有効な行が無く保存されなかった銘柄は error になり、成功に数えないこと（#1293）"""
        from src.backtest.optimizer import run_optuna_batch
        from src.backtest.optimizer.persistence import save_optimal_params_json as real_save_json
        from src.domain.types import SymbolTask

        error_df = pd.DataFrame(
            {"threshold": [0.01, 0.02], "error": ["データが取得できませんでした"] * 2}
        )
        good_df = pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5]})

        def _json(df, market, symbol, sort_by="sharpe_ratio"):
            if symbol == "7203":
                # 有効な行が無い表では、本物は何も書かずに "" を返す
                return real_save_json(df, market, symbol, sort_by=sort_by)
            return "/tmp/params.json"

        mock_run.side_effect = lambda market, symbol, **kwargs: (
            error_df if symbol == "7203" else good_df
        )
        mock_json.side_effect = _json

        results = run_optuna_batch(
            [SymbolTask(market="jp", symbol="7203"), SymbolTask(market="jp", symbol="9984")],
            n_trials=1,
            max_workers=1,
        )

        by_symbol = {r["symbol"]: r for r in results}
        self.assertEqual(
            by_symbol["7203"],
            {"market": "jp", "symbol": "7203", "status": "error", "error": "有効な最適化結果なし"},
        )
        self.assertEqual(by_symbol["9984"], {"market": "jp", "symbol": "9984", "status": "success"})


# ──────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio による選択バイアスガード（#467 / #372）
# ──────────────────────────────────────────────────────────────────


class TestOptunaDSRGuard(unittest.TestCase):
    """Optuna 最適化が最良戦略に DSR を付与することの検証。"""

    def _make_wf_df(self):
        return pd.DataFrame(
            {
                "sharpe_ratio": [1.5],
                "num_trades": [50],
            }
        )

    @patch("src.backtest.optimizer.optuna_search.run_backtest_walk_forward")
    def test_best_row_has_dsr_and_num_trades(self, mock_wf):
        from src.backtest.optimizer import run_optuna_optimization

        mock_wf.return_value = (None, None, self._make_wf_df())

        result = run_optuna_optimization(
            market="jp",
            symbol="7203",
            n_trials=3,
        )
        # num_trades 列が全行に付与される
        self.assertIn("num_trades", result.columns)
        self.assertTrue((result["num_trades"] == 50).all())
        # DSR は最良戦略の1行だけに付与される
        self.assertIn("dsr", result.columns)
        non_na = result["dsr"].dropna()
        self.assertEqual(len(non_na), 1)
        dsr_value = float(non_na.iloc[0])
        self.assertGreaterEqual(dsr_value, 0.0)
        self.assertLessEqual(dsr_value, 1.0)

    @patch("src.backtest.optimizer.optuna_search.run_backtest_walk_forward")
    def test_dsr_uses_trial_count_for_correction(self, mock_wf):
        """試行数が増えるほど多重比較補正で DSR は上がりにくくなる（単調性の sanity）。"""
        from src.backtest.optimizer import run_optuna_optimization

        mock_wf.return_value = (None, None, self._make_wf_df())

        few = run_optuna_optimization(market="jp", symbol="7203", n_trials=2)
        many = run_optuna_optimization(market="jp", symbol="7203", n_trials=20)
        dsr_few = float(few["dsr"].dropna().iloc[0])
        dsr_many = float(many["dsr"].dropna().iloc[0])
        # 同じ Sharpe でも試行数が多いほど期待最大値が上がり DSR は下がる
        self.assertGreaterEqual(dsr_few, dsr_many)

    @staticmethod
    def _wf_by_threshold(**kwargs):
        """閾値が大きいほど DD が深く・Sharpe が高く・コストが大きい fold 結果を返す。"""
        th = float(kwargs["threshold"])
        return (
            None,
            None,
            pd.DataFrame(
                {
                    "max_drawdown": [-0.01 - th * 10],
                    "sharpe_ratio": [th * 100],
                    "cost_impact_return": [th],
                    "num_trades": [10],
                }
            ),
        )

    def _best_row(self, sort_by: str):
        from src.backtest.optimizer import run_optuna_optimization

        result = run_optuna_optimization(market="jp", symbol="7203", n_trials=10, sort_by=sort_by)
        return result.loc[result["dsr"].notna()].iloc[0], result

    @patch("src.backtest.optimizer.optuna_search.run_backtest_walk_forward")
    def test_max_drawdown_is_maximized(self, mock_wf):
        """max_drawdown は負の小数なので 0 に近い（大きい）組が最良になる。"""
        mock_wf.side_effect = self._wf_by_threshold
        best, result = self._best_row("max_drawdown")
        self.assertEqual(float(best["max_drawdown"]), float(result["max_drawdown"].max()))

    @patch("src.backtest.optimizer.optuna_search.run_backtest_walk_forward")
    def test_sharpe_ratio_is_maximized(self, mock_wf):
        mock_wf.side_effect = self._wf_by_threshold
        best, result = self._best_row("sharpe_ratio")
        self.assertEqual(float(best["sharpe_ratio"]), float(result["sharpe_ratio"].max()))

    @patch("src.backtest.optimizer.optuna_search.run_backtest_walk_forward")
    def test_cost_impact_return_is_minimized(self, mock_wf):
        mock_wf.side_effect = self._wf_by_threshold
        best, result = self._best_row("cost_impact_return")
        self.assertEqual(
            float(best["cost_impact_return"]), float(result["cost_impact_return"].min())
        )


class TestSaveOptimalParamsDSR(unittest.TestCase):
    """save_optimal_params_json が DSR を JSON metrics に記録することの検証。"""

    def _best_df(self, with_dsr: bool):
        data = {
            "threshold": [0.01],
            "sharpe_ratio": [1.5],
            "num_trades": [50],
        }
        if with_dsr:
            data["dsr"] = [0.87]
        return pd.DataFrame(data)

    def _save_and_capture(self, df):
        """save_optimal_params_json を実ファイルに触れず実行し、書き込まれた dict を返す。"""
        return _save_and_read(df)

    def test_records_dsr_when_present(self):
        saved = self._save_and_capture(self._best_df(with_dsr=True))
        metrics = saved["jp_7203"]["metrics"]
        self.assertAlmostEqual(metrics["deflated_sharpe_ratio"], 0.87, places=6)

    def test_dsr_none_when_absent(self):
        saved = self._save_and_capture(self._best_df(with_dsr=False))
        self.assertIsNone(saved["jp_7203"]["metrics"]["deflated_sharpe_ratio"])


# ──────────────────────────────────────────────────────────────────
# PBO (Probability of Backtest Overfitting) ガード配線（#372）
# ──────────────────────────────────────────────────────────────────


class TestComputePboFromFoldReturns(unittest.TestCase):
    """_compute_pbo_from_fold_returns の純粋関数テスト"""

    def test_nan_when_less_than_two_candidates(self):
        """候補が2未満のとき NaN"""
        import math

        from src.backtest.optimizer import _compute_pbo_from_fold_returns

        self.assertTrue(math.isnan(_compute_pbo_from_fold_returns([], "jp", "7203")))
        self.assertTrue(math.isnan(_compute_pbo_from_fold_returns([[0.1, 0.2, 0.3]], "jp", "7203")))

    def test_nan_when_folds_too_short(self):
        """fold 数が2未満の系列しかないとき NaN"""
        import math

        from src.backtest.optimizer import _compute_pbo_from_fold_returns

        self.assertTrue(math.isnan(_compute_pbo_from_fold_returns([[0.1], [0.2]], "jp", "7203")))

    def test_returns_value_in_unit_interval(self):
        """有効な入力で 0〜1 の値が返ること"""
        import numpy as np

        from src.backtest.optimizer import _compute_pbo_from_fold_returns

        rng = np.random.default_rng(42)
        fold_returns = [rng.normal(0, 0.01, size=8).tolist() for _ in range(6)]
        pbo = _compute_pbo_from_fold_returns(fold_returns, "jp", "7203")
        self.assertGreaterEqual(pbo, 0.0)
        self.assertLessEqual(pbo, 1.0)

    def test_truncates_unequal_length_series(self):
        """系列長が揃っていなくても最短長に揃えて計算されること"""
        import math

        import numpy as np

        from src.backtest.optimizer import _compute_pbo_from_fold_returns

        rng = np.random.default_rng(7)
        fold_returns = [
            rng.normal(0, 0.01, size=8).tolist(),
            rng.normal(0, 0.01, size=6).tolist(),
            rng.normal(0, 0.01, size=7).tolist(),
        ]
        pbo = _compute_pbo_from_fold_returns(fold_returns, "jp", "7203")
        self.assertFalse(math.isnan(pbo))


class TestRunOptimizationPBO(unittest.TestCase):
    """run_optimization が PBO 列を付与することの検証。"""

    def _make_wf_df(self, returns):
        return pd.DataFrame(
            {
                "total_return": returns,
                "sharpe_ratio": [1.0] * len(returns),
                "num_trades": [10] * len(returns),
            }
        )

    @patch("src.backtest.optimizer.grid.run_backtest_walk_forward")
    def test_pbo_column_attached_with_multiple_candidates(self, mock_wf):
        """複数候補 × 複数 fold で pbo 列が全行に同一値で付与されること"""
        from src.backtest.optimizer import run_optimization

        mock_wf.side_effect = [
            (None, None, self._make_wf_df([0.05, -0.02, 0.03, 0.01])),
            (None, None, self._make_wf_df([0.01, 0.04, -0.01, 0.02])),
        ]
        result = run_optimization(
            market="jp",
            symbol="7203",
            threshold_min=0.01,
            threshold_max=0.02,
            threshold_step=0.01,
        )
        self.assertIn("pbo", result.columns)
        values = result["pbo"].dropna().unique()
        self.assertEqual(len(values), 1)
        self.assertGreaterEqual(float(values[0]), 0.0)
        self.assertLessEqual(float(values[0]), 1.0)

    @patch("src.backtest.optimizer.grid.run_backtest_walk_forward")
    def test_no_pbo_column_with_single_candidate(self, mock_wf):
        """候補が1つだけのとき pbo 列は付与されないこと"""
        from src.backtest.optimizer import run_optimization

        mock_wf.return_value = (None, None, self._make_wf_df([0.05, -0.02, 0.03, 0.01]))
        result = run_optimization(
            market="jp",
            symbol="7203",
            threshold_min=0.01,
            threshold_max=0.01,
            threshold_step=0.01,
        )
        self.assertNotIn("pbo", result.columns)


class TestOptunaPBOGuard(unittest.TestCase):
    """Optuna 最適化が PBO を全行に付与することの検証。"""

    def _make_wf_df(self):
        return pd.DataFrame(
            {
                "total_return": [0.05, -0.02, 0.03, 0.01],
                "sharpe_ratio": [1.5, -0.5, 1.0, 0.3],
                "num_trades": [10, 12, 8, 15],
            }
        )

    @patch("src.backtest.optimizer.optuna_search.run_backtest_walk_forward")
    def test_all_rows_have_pbo(self, mock_wf):
        from src.backtest.optimizer import run_optuna_optimization

        mock_wf.return_value = (None, None, self._make_wf_df())

        result = run_optuna_optimization(market="jp", symbol="7203", n_trials=3)
        self.assertIn("pbo", result.columns)
        non_na = result["pbo"].dropna()
        self.assertEqual(len(non_na), len(result))
        self.assertEqual(len(non_na.unique()), 1)
        self.assertGreaterEqual(float(non_na.iloc[0]), 0.0)
        self.assertLessEqual(float(non_na.iloc[0]), 1.0)


class TestSaveOptimalParamsPBO(unittest.TestCase):
    """save_optimal_params_json が PBO を JSON metrics に記録することの検証。"""

    def _best_df(self, with_pbo: bool):
        data = {
            "threshold": [0.01],
            "sharpe_ratio": [1.5],
            "num_trades": [50],
        }
        if with_pbo:
            data["pbo"] = [0.32]
        return pd.DataFrame(data)

    def _save_and_capture(self, df):
        return _save_and_read(df)

    def test_records_pbo_when_present(self):
        saved = self._save_and_capture(self._best_df(with_pbo=True))
        self.assertAlmostEqual(saved["jp_7203"]["metrics"]["pbo"], 0.32, places=6)

    def test_pbo_none_when_absent(self):
        saved = self._save_and_capture(self._best_df(with_pbo=False))
        self.assertIsNone(saved["jp_7203"]["metrics"]["pbo"])


if __name__ == "__main__":
    unittest.main()
