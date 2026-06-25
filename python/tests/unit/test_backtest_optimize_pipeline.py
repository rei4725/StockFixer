"""Unit tests for backtest_optimize_pipeline.py"""

import json
import os
import tempfile
import unittest
from unittest.mock import mock_open, patch

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
                "src.backtest.optimizer.persistence.get_results_dir",
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
                "src.backtest.optimizer.persistence.get_results_dir",
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

        from src.backtest.optimizer import save_optimal_params_json

        m = mopen()
        with patch("os.path.exists", return_value=False), patch(
            "src.backtest.optimizer.persistence.ensure_dir"
        ), patch("builtins.open", m):
            path = save_optimal_params_json(
                self._make_result_df(), "jp", "7203", sort_by="sharpe_ratio"
            )

        self.assertIsNotNone(path)
        self.assertNotEqual(path, "")
        self.assertTrue(path.endswith("optimal_params.json"))

    def test_returns_empty_on_empty_df(self):
        """DataFrame が空の場合、空文字列が返ること"""
        from src.backtest.optimizer import save_optimal_params_json

        result = save_optimal_params_json(pd.DataFrame(), "jp", "7203")
        self.assertEqual(result, "")

    def test_merges_with_existing_json(self):
        """既存 JSON と結果が統合されること"""
        import json as _json

        from src.backtest.optimizer import save_optimal_params_json

        existing_data = {"jp_AAPL": {"threshold": 0.01}}
        captured: dict = {}

        def fake_dump(obj, fp, **kwargs):
            captured["obj"] = obj

        m = mock_open(read_data=_json.dumps(existing_data))
        with patch("os.path.exists", return_value=True), patch(
            "src.backtest.optimizer.persistence.ensure_dir"
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

    def test_returns_empty_on_no_tasks(self):
        """タスクが空の場合、空リストが返ること"""
        from src.backtest.optimizer import run_optimize_batch

        results = run_optimize_batch([], model_type="XGBoostModel")
        self.assertEqual(results, [])


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
        from src.backtest.optimizer import save_optimal_params_json

        with patch("src.backtest.optimizer.persistence.os.path.exists", return_value=False), patch(
            "src.backtest.optimizer.persistence.ensure_dir"
        ), patch("builtins.open", mock_open()), patch(
            "src.backtest.optimizer.persistence.json.dump"
        ) as mock_dump:
            save_optimal_params_json(df, "jp", "7203")
        return mock_dump.call_args[0][0]

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
        from src.backtest.optimizer import save_optimal_params_json

        with patch("src.backtest.optimizer.persistence.os.path.exists", return_value=False), patch(
            "src.backtest.optimizer.persistence.ensure_dir"
        ), patch("builtins.open", mock_open()), patch(
            "src.backtest.optimizer.persistence.json.dump"
        ) as mock_dump:
            save_optimal_params_json(df, "jp", "7203")
        return mock_dump.call_args[0][0]

    def test_records_pbo_when_present(self):
        saved = self._save_and_capture(self._best_df(with_pbo=True))
        self.assertAlmostEqual(saved["jp_7203"]["metrics"]["pbo"], 0.32, places=6)

    def test_pbo_none_when_absent(self):
        saved = self._save_and_capture(self._best_df(with_pbo=False))
        self.assertIsNone(saved["jp_7203"]["metrics"]["pbo"])


if __name__ == "__main__":
    unittest.main()
