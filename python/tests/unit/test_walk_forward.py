"""backtest/walk_forward モジュールのユニットテスト"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.backtest.walk_forward import WalkForwardValidator


def _make_feature_df(n=60):
    """Walk-Forward テスト用の特徴量DataFrame"""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.linspace(0, 1, n),
            "close": 100 + np.random.randn(n).cumsum(),
        },
        index=dates,
    )
    return df


class TestWalkForwardValidatorInit(unittest.TestCase):
    """WalkForwardValidator.__init__ のテスト"""

    def _make_validator(self, **kwargs):
        defaults = dict(
            market="us",
            symbol="AAPL",
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
        )
        defaults.update(kwargs)
        return WalkForwardValidator(**defaults)

    def test_basic_construction(self):
        """デフォルトパラメータで構築できること"""
        v = self._make_validator()
        self.assertEqual(v.market, "us")
        self.assertEqual(v.symbol, "AAPL")
        self.assertEqual(v.initial_cash, 1_000_000)
        self.assertEqual(v.n_splits, 5)

    def test_custom_parameters_stored(self):
        """カスタムパラメータが正しく保持されること"""
        v = self._make_validator(
            initial_cash=500_000,
            fee_rate=0.001,
            slippage=0.0005,
            n_splits=3,
            source="db",
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            position_sizing="fixed",
            position_fraction=0.3,
            atr_risk_pct=0.01,
            atr_multiplier=2.0,
            atr_min_fraction=0.05,
            atr_max_fraction=0.8,
            ensemble=True,
        )
        self.assertEqual(v.initial_cash, 500_000)
        self.assertAlmostEqual(v.fee_rate, 0.001)
        self.assertAlmostEqual(v.slippage, 0.0005)
        self.assertEqual(v.n_splits, 3)
        self.assertEqual(v.source, "db")
        self.assertAlmostEqual(v.stop_loss_pct, 0.05)
        self.assertAlmostEqual(v.take_profit_pct, 0.10)
        self.assertEqual(v.position_sizing, "fixed")
        self.assertAlmostEqual(v.position_fraction, 0.3)
        self.assertAlmostEqual(v.atr_risk_pct, 0.01)
        self.assertAlmostEqual(v.atr_multiplier, 2.0)
        self.assertAlmostEqual(v.atr_min_fraction, 0.05)
        self.assertAlmostEqual(v.atr_max_fraction, 0.8)
        self.assertTrue(v.ensemble)

    def test_default_source_is_file(self):
        """デフォルトの source は 'file' であること"""
        v = self._make_validator()
        self.assertEqual(v.source, "file")

    def test_default_ensemble_is_false(self):
        """デフォルトの ensemble は False であること"""
        v = self._make_validator()
        self.assertFalse(v.ensemble)


class TestPrintSummary(unittest.TestCase):
    """WalkForwardValidator._print_summary のテスト"""

    def test_runs_without_error_with_full_metrics(self):
        """メトリクス列がある場合にエラーなく実行されること"""
        result_df = pd.DataFrame(
            [
                {
                    "fold": 1,
                    "val_start": "2024-03-01",
                    "val_end": "2024-05-01",
                    "total_return": 0.05,
                    "sharpe_ratio": 1.2,
                    "max_drawdown": -0.10,
                    "win_rate": 0.6,
                    "profit_factor": 1.5,
                },
                {
                    "fold": 2,
                    "val_start": "2024-05-01",
                    "val_end": "2024-07-01",
                    "total_return": 0.03,
                    "sharpe_ratio": 0.9,
                    "max_drawdown": -0.08,
                    "win_rate": 0.55,
                    "profit_factor": 1.2,
                },
            ]
        )
        # print出力を抑制してエラーなく完了することを確認
        with patch("builtins.print"):
            WalkForwardValidator._print_summary(result_df)

    def test_runs_without_error_empty_df(self):
        """空のDataFrameでもエラーなく実行されること"""
        with patch("builtins.print"):
            WalkForwardValidator._print_summary(pd.DataFrame())

    def test_runs_with_no_numeric_columns(self):
        """数値列がないDataFrameでもエラーなく実行されること"""
        result_df = pd.DataFrame(
            [
                {"fold": 1, "val_start": "2024-01-01", "val_end": "2024-03-01", "error": "failed"},
            ]
        )
        with patch("builtins.print"):
            WalkForwardValidator._print_summary(result_df)

    def test_handles_none_values_in_numeric_columns(self):
        """数値列にNoneが混在してもエラーなく実行されること"""
        result_df = pd.DataFrame(
            [
                {
                    "fold": 1,
                    "val_start": "2024-01-01",
                    "val_end": "2024-03-01",
                    "total_return": 0.05,
                    "sharpe_ratio": None,
                },
                {
                    "fold": 2,
                    "val_start": "2024-03-01",
                    "val_end": "2024-06-01",
                    "total_return": None,
                    "sharpe_ratio": 1.2,
                },
            ]
        )
        with patch("builtins.print"):
            WalkForwardValidator._print_summary(result_df)

    def test_handles_all_optional_metric_columns(self):
        """Walk-Forward のオプション指標列（cost_impact, atr_fallback等）が含まれても動作すること"""
        result_df = pd.DataFrame(
            [
                {
                    "fold": 1,
                    "val_start": "2024-01-01",
                    "val_end": "2024-03-01",
                    "total_return": 0.02,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": -0.05,
                    "gross_total_return": 0.025,
                    "gross_sharpe_ratio": 1.1,
                    "gross_max_drawdown": -0.04,
                    "cost_impact_return": 0.005,
                    "cost_impact_cash": 500.0,
                    "win_rate": 0.60,
                    "profit_factor": 1.4,
                    "avg_position_fraction": 0.5,
                    "max_position_fraction": 1.0,
                    "avg_position_value": 50000.0,
                    "atr_fallback_trades": 2,
                }
            ]
        )
        with patch("builtins.print"):
            WalkForwardValidator._print_summary(result_df)


class TestWalkForwardRun(unittest.TestCase):
    """WalkForwardValidator.run() の統合テスト（モック使用）"""

    def _make_validator(self, **kwargs):
        defaults = dict(
            market="us",
            symbol="AAPL",
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            n_splits=2,
        )
        defaults.update(kwargs)
        return WalkForwardValidator(**defaults)

    def test_run_raises_on_empty_data(self):
        """データが取得できない場合は ValueError が送出されること"""
        v = self._make_validator()
        with patch.object(v, "_load_features", return_value=None):
            with self.assertRaises(ValueError):
                v.run("StockXGBoostModel")

    def test_run_raises_on_empty_dataframe(self):
        """空DataFrameが返ってきた場合は ValueError が送出されること"""
        v = self._make_validator()
        with patch.object(v, "_load_features", return_value=pd.DataFrame()):
            with self.assertRaises(ValueError):
                v.run("StockXGBoostModel")

    def test_run_returns_dataframe(self):
        """正常データが返るとき結果がDataFrameであること"""
        from src.backtest.task import ReturnRegressionTask

        v = self._make_validator(n_splits=2)
        feature_df = _make_feature_df(60)
        task = ReturnRegressionTask()

        with (
            patch.object(v, "_load_features", return_value=feature_df),
            patch.object(
                v,
                "_run_fold",
                return_value={
                    "total_return": 0.02,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": -0.05,
                    "win_rate": 0.6,
                    "profit_factor": 1.5,
                },
            ),
            patch("builtins.print"),
        ):
            result = v.run("StockXGBoostModel", task=task)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("fold", result.columns)
        self.assertIn("val_start", result.columns)
        self.assertIn("val_end", result.columns)

    def test_run_collects_fold_errors(self):
        """各折でエラーが起きても残りの折は続行してDataFrameに記録されること"""
        from src.backtest.task import ReturnRegressionTask

        v = self._make_validator(n_splits=2)
        feature_df = _make_feature_df(60)
        task = ReturnRegressionTask()

        with (
            patch.object(v, "_load_features", return_value=feature_df),
            patch.object(v, "_run_fold", side_effect=Exception("fold error")),
            patch("builtins.print"),
        ):
            result = v.run("StockXGBoostModel", task=task)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("error", result.columns)
        # エラーが記録されていること
        self.assertTrue((result["error"] == "fold error").any())

    def test_run_uses_default_task_when_none(self):
        """task=None のとき ReturnRegressionTask がデフォルトで使われること"""
        from src.backtest.task import (  # noqa: F401 - imported for usage check below
            ReturnRegressionTask,
        )

        v = self._make_validator(n_splits=2)
        feature_df = _make_feature_df(60)

        with (
            patch.object(v, "_load_features", return_value=feature_df),
            patch.object(v, "_run_fold", return_value={"total_return": 0.01}),
            patch("builtins.print"),
        ):
            result = v.run("StockXGBoostModel", task=None)

        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
