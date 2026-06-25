"""
R-305: セクターローテーション戦略のユニットテスト
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class TestGetRegimeSectorWeight(unittest.TestCase):
    """get_regime_sector_weight のテスト"""

    def setUp(self):
        from src.trading.signal_generator import get_regime_sector_weight

        self.fn = get_regime_sector_weight

    def test_bull_momentum_sector_returns_high_weight(self):
        """bull レジームのモメンタム系セクターは高ウェイトを返す"""
        w = self.fn("bull", "Technology")
        self.assertGreater(w, 1.0)

    def test_bull_defensive_sector_returns_low_weight(self):
        """bull レジームのディフェンシブ系セクターは抑制ウェイトを返す"""
        w = self.fn("bull", "Utilities")
        self.assertLess(w, 1.0)

    def test_bear_defensive_sector_returns_high_weight(self):
        """bear レジームのディフェンシブ系セクターは高ウェイトを返す"""
        w = self.fn("bear", "Healthcare")
        self.assertGreater(w, 1.0)

    def test_bear_momentum_sector_returns_low_weight(self):
        """bear レジームのモメンタム系セクターは抑制ウェイトを返す"""
        w = self.fn("bear", "Technology")
        self.assertLess(w, 1.0)

    def test_range_returns_uniform_weight(self):
        """range レジームは全セクターが 1.0 を返す"""
        for sector in ("Technology", "Healthcare", "Utilities", "UnknownSector"):
            with self.subTest(sector=sector):
                self.assertAlmostEqual(self.fn("range", sector), 1.0)

    def test_unknown_regime_returns_uniform_weight(self):
        """未知レジームは 1.0 を返す"""
        self.assertAlmostEqual(self.fn("unknown", "Technology"), 1.0)

    def test_returns_float(self):
        """戻り値が float であること"""
        self.assertIsInstance(self.fn("bull", "Technology"), float)


class TestRegimeSectorWeightsConstants(unittest.TestCase):
    """REGIME_SECTOR_WEIGHTS 定数のテスト"""

    def test_bull_and_bear_keys_exist(self):
        from src.trading.signal_generator import REGIME_SECTOR_WEIGHTS

        self.assertIn("bull", REGIME_SECTOR_WEIGHTS)
        self.assertIn("bear", REGIME_SECTOR_WEIGHTS)
        self.assertIn("range", REGIME_SECTOR_WEIGHTS)

    def test_bull_has_momentum_sectors(self):
        from src.trading.signal_generator import REGIME_SECTOR_WEIGHTS

        self.assertIn("Technology", REGIME_SECTOR_WEIGHTS["bull"])
        self.assertIn("Consumer Cyclical", REGIME_SECTOR_WEIGHTS["bull"])

    def test_bear_has_defensive_sectors(self):
        from src.trading.signal_generator import REGIME_SECTOR_WEIGHTS

        self.assertIn("Healthcare", REGIME_SECTOR_WEIGHTS["bear"])
        self.assertIn("Utilities", REGIME_SECTOR_WEIGHTS["bear"])

    def test_range_is_empty_dict(self):
        from src.trading.signal_generator import REGIME_SECTOR_WEIGHTS

        self.assertEqual(REGIME_SECTOR_WEIGHTS["range"], {})


class TestApplySectorRotation(unittest.TestCase):
    """_apply_sector_rotation のテスト"""

    @patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector")
    def test_multiplies_scores_by_regime_weight(self, mock_sector):
        """レジームウェイトがスコアに乗算されること"""
        from src.backtest.portfolio import _apply_sector_rotation

        mock_sector.return_value = "Technology"
        scores = pd.Series({"jp_7203": 0.05, "jp_6758": 0.03})
        result = _apply_sector_rotation(scores, "bull")

        # Technology は bull で 2.0 倍
        self.assertAlmostEqual(result["jp_7203"], 0.05 * 2.0)
        self.assertAlmostEqual(result["jp_6758"], 0.03 * 2.0)

    @patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector")
    def test_range_regime_does_not_change_scores(self, mock_sector):
        """range レジームではスコアが変わらないこと（乗数 1.0）"""
        from src.backtest.portfolio import _apply_sector_rotation

        mock_sector.return_value = "Technology"
        scores = pd.Series({"jp_7203": 0.04})
        result = _apply_sector_rotation(scores, "range")
        self.assertAlmostEqual(result["jp_7203"], 0.04)

    @patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector")
    def test_does_not_mutate_original_series(self, mock_sector):
        """元の Series を変更しないこと"""
        from src.backtest.portfolio import _apply_sector_rotation

        mock_sector.return_value = "Utilities"
        scores = pd.Series({"jp_9503": 0.02})
        original_val = scores["jp_9503"]
        _apply_sector_rotation(scores, "bull")
        self.assertAlmostEqual(scores["jp_9503"], original_val)


class TestSimulatePortfolioWithSectorRotation(unittest.TestCase):
    """_simulate_portfolio に use_sector_rotation=True を渡したときのテスト"""

    def _make_matrices(self, symbols, n_days=15):
        dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
        rng = np.random.default_rng(seed=0)
        score_data = {sym: rng.random(n_days) * 0.1 + 0.01 for sym in symbols}
        close_data = {sym: rng.random(n_days) * 100 + 100 for sym in symbols}
        return pd.DataFrame(score_data, index=dates), pd.DataFrame(close_data, index=dates)

    @patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector")
    def test_returns_equity_curve_with_rotation(self, mock_sector):
        """sector_rotation=True でも equity_df が返ること"""
        from src.backtest.portfolio import _get_rebalance_dates, _simulate_portfolio

        mock_sector.return_value = "Technology"
        symbols = ["jp_7203", "jp_6758"]
        score_matrix, close_matrix = self._make_matrices(symbols)
        rebalance_dates = _get_rebalance_dates(score_matrix.index, "weekly")

        eq_df, holdings = _simulate_portfolio(
            score_matrix,
            close_matrix,
            rebalance_dates,
            top_n=2,
            initial_cash=1_000_000,
            fee_rate=0.001,
            max_sector_positions=3,
            use_sector_rotation=True,
        )
        self.assertIn("portfolio_value", eq_df.columns)
        self.assertGreater(len(eq_df), 0)

    @patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector")
    def test_rotation_on_off_both_valid(self, mock_sector):
        """rotation ON/OFF どちらも有効な equity_df を返すこと"""
        from src.backtest.portfolio import _get_rebalance_dates, _simulate_portfolio

        mock_sector.return_value = "Healthcare"
        symbols = ["jp_7203", "jp_6758", "jp_9503"]
        score_matrix, close_matrix = self._make_matrices(symbols)
        rebalance_dates = _get_rebalance_dates(score_matrix.index, "weekly")

        eq_off, _ = _simulate_portfolio(
            score_matrix,
            close_matrix,
            rebalance_dates,
            top_n=2,
            initial_cash=1_000_000,
            fee_rate=0.001,
            max_sector_positions=3,
            use_sector_rotation=False,
        )
        eq_on, _ = _simulate_portfolio(
            score_matrix,
            close_matrix,
            rebalance_dates,
            top_n=2,
            initial_cash=1_000_000,
            fee_rate=0.001,
            max_sector_positions=3,
            use_sector_rotation=True,
        )
        self.assertEqual(len(eq_off), len(eq_on))
        self.assertIn("portfolio_value", eq_on.columns)


class TestCompareSectorRotationKpi(unittest.TestCase):
    """compare_sector_rotation_kpi のテスト"""

    @patch("src.utils.db.stock_features.get_all_symbols")
    def test_returns_empty_on_no_symbols(self, mock_symbols):
        from src.backtest.portfolio import compare_sector_rotation_kpi

        mock_symbols.return_value = []
        result = compare_sector_rotation_kpi(market="jp")
        self.assertEqual(result, {})

    @patch("src.backtest.portfolio._build_signal_matrix")
    @patch("src.utils.db.stock_features.get_all_symbols")
    def test_returns_empty_on_empty_score_matrix(self, mock_symbols, mock_build):
        from src.backtest.portfolio import compare_sector_rotation_kpi

        mock_symbols.return_value = [("jp", "7203")]
        mock_build.return_value = (pd.DataFrame(), pd.DataFrame())
        result = compare_sector_rotation_kpi(market="jp")
        self.assertEqual(result, {})

    @patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector")
    @patch("src.backtest.portfolio._build_signal_matrix")
    @patch("src.utils.db.stock_features.get_all_symbols")
    def test_returns_comparison_dict_with_expected_keys(
        self, mock_symbols, mock_build, mock_sector
    ):
        from src.backtest.portfolio import compare_sector_rotation_kpi

        mock_symbols.return_value = [("jp", "7203"), ("jp", "6758")]
        mock_sector.return_value = "Technology"

        n = 20
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        rng = np.random.default_rng(seed=1)
        score_matrix = pd.DataFrame(
            {"jp_7203": rng.random(n) * 0.1, "jp_6758": rng.random(n) * 0.1}, index=dates
        )
        close_matrix = pd.DataFrame(
            {"jp_7203": rng.random(n) * 100 + 100, "jp_6758": rng.random(n) * 100 + 100},
            index=dates,
        )
        mock_build.return_value = (score_matrix, close_matrix)

        result = compare_sector_rotation_kpi(market="jp", rebalance_freq="weekly")

        self.assertIn("rotation_off", result)
        self.assertIn("rotation_on", result)
        self.assertIn("kpi_diff", result)
        self.assertIn("total_return_diff", result["kpi_diff"])
        self.assertIn("sharpe_diff", result["kpi_diff"])
        self.assertIn("max_drawdown_diff", result["kpi_diff"])

    @patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector")
    @patch("src.backtest.portfolio._build_signal_matrix")
    @patch("src.utils.db.stock_features.get_all_symbols")
    def test_kpi_diff_values_are_numeric(self, mock_symbols, mock_build, mock_sector):
        """kpi_diff の値が数値であること"""
        from src.backtest.portfolio import compare_sector_rotation_kpi

        mock_symbols.return_value = [("jp", "7203")]
        mock_sector.return_value = "Healthcare"

        n = 20
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        rng = np.random.default_rng(seed=2)
        score_matrix = pd.DataFrame({"jp_7203": rng.random(n) * 0.05 + 0.01}, index=dates)
        close_matrix = pd.DataFrame({"jp_7203": rng.random(n) * 100 + 100}, index=dates)
        mock_build.return_value = (score_matrix, close_matrix)

        result = compare_sector_rotation_kpi(market="jp", rebalance_freq="weekly")
        if result:
            for k, v in result["kpi_diff"].items():
                with self.subTest(key=k):
                    self.assertIsInstance(v, float)


if __name__ == "__main__":
    unittest.main()
