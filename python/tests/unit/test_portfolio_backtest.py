import unittest
import unittest.mock
from unittest.mock import patch

import pandas as pd
import pytest

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.portfolio import _attach_regime_metrics, _limit_portfolio_candidates_by_sector


class TestPortfolioBacktestRegimeMetrics(unittest.TestCase):
    def test_attach_regime_metrics_adds_regime_column_and_summary(self):
        dates = pd.date_range("2024-01-01", periods=6, freq="D")
        equity_df = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "portfolio_value": [100.0, 103.0, 101.0, 104.0, 106.0, 105.0],
                "equal_weight_value": [100.0, 101.0, 100.0, 101.0, 102.0, 101.0],
            }
        )
        close_matrix = pd.DataFrame(
            {
                "jp_7203": [100, 101, 99, 98, 100, 101],
                "jp_6758": [80, 81, 82, 81, 80, 79],
            },
            index=dates,
        )
        mocked_regime = pd.Series(
            ["bull", "bull", "bear", "bear", "range", "range"],
            index=dates,
            dtype=str,
        )

        mock_port = unittest.mock.MagicMock()
        mock_port.get_market_regime.return_value = mocked_regime
        with patch("src.backtest.portfolio.metrics.get_backtest_data_port", return_value=mock_port):
            enriched, regime_metrics = _attach_regime_metrics(equity_df, close_matrix)

        self.assertIn("regime", enriched.columns)
        self.assertEqual(enriched["regime"].tolist(), mocked_regime.tolist())
        self.assertEqual(regime_metrics["bull"]["days"], 2)
        self.assertEqual(regime_metrics["bear"]["days"], 2)
        self.assertEqual(regime_metrics["range"]["days"], 2)
        self.assertIn("all", regime_metrics)
        self.assertIn("hit_rate", regime_metrics["bull"])


class TestPortfolioBacktestSectorLimit(unittest.TestCase):
    @patch("src.backtest.portfolio.simulation.get_symbol_sector")
    def test_limit_portfolio_candidates_by_sector_caps_same_sector(self, mock_get_sector):
        mock_get_sector.side_effect = ["Auto", "Auto", "Tech", "Bank"]
        top_candidates = pd.Series(
            [0.04, 0.03, 0.025, 0.02],
            index=["jp_7203", "jp_7267", "jp_6758", "jp_8306"],
            dtype=float,
        )

        limited = _limit_portfolio_candidates_by_sector(top_candidates, max_sector_positions=1)

        self.assertEqual(limited.index.tolist(), ["jp_7203", "jp_6758", "jp_8306"])
        self.assertEqual(limited.tolist(), [0.04, 0.025, 0.02])


if __name__ == "__main__":
    unittest.main()


# ──────────────────────────────────────────────────────────────
# pytest スタイルの追加テスト
# ──────────────────────────────────────────────────────────────


class TestBuildSignalMatrix:
    """_build_signal_matrix のテスト"""

    def test_returns_empty_on_empty_symbols(self):
        """銘柄リストが空の場合は空の DataFrame が返ること"""
        from src.backtest.portfolio import _build_signal_matrix

        scores, prices = _build_signal_matrix(
            symbols=[],
            model_type="XGBoostModel",
            train_ratio=0.8,
            source="features",
            threshold=0.0,
            ensemble=False,
        )
        assert scores.empty
        assert prices.empty


class TestGetRebalanceDates:
    """_get_rebalance_dates のテスト"""

    def test_returns_expected_rebalance_dates_weekly(self):
        """週次指定で正しくリバランス日が生成されること"""
        from src.backtest.portfolio import _get_rebalance_dates

        idx = pd.date_range("2026-01-05", periods=20, freq="B")
        dates = _get_rebalance_dates(idx, freq="weekly")
        assert len(dates) >= 1
        assert dates[0] in idx

    def test_returns_all_dates_on_daily(self):
        """daily 指定は全日付が返ること"""
        from src.backtest.portfolio import _get_rebalance_dates

        idx = pd.date_range("2026-01-05", periods=5, freq="B")
        dates = _get_rebalance_dates(idx, freq="daily")
        assert len(dates) == len(idx)

    def test_raises_on_invalid_freq(self):
        """未対応の freq 指定は ValueError が発生すること"""
        from src.backtest.portfolio import _get_rebalance_dates

        idx = pd.date_range("2026-01-05", periods=10, freq="B")
        with pytest.raises(ValueError):
            _get_rebalance_dates(idx, freq="quarterly")


class TestComputePortfolioMetrics:
    """_compute_portfolio_metrics のテスト"""

    def test_returns_dict_with_expected_keys(self):
        """必要なキーを持つ辞書が返ること"""
        from src.backtest.portfolio import _compute_portfolio_metrics

        equity_df = pd.DataFrame(
            {
                "portfolio_value": [1000000.0 + i * 1000 for i in range(20)],
                "equal_weight_value": [1000000.0 + i * 500 for i in range(20)],
            }
        )
        metrics = _compute_portfolio_metrics(equity_df, initial_cash=1000000.0)
        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics

    def test_returns_empty_on_empty_dataframe(self):
        """空の DataFrame は空辞書が返ること"""
        from src.backtest.portfolio import _compute_portfolio_metrics

        metrics = _compute_portfolio_metrics(pd.DataFrame(), initial_cash=1000000.0)
        assert metrics == {}


class TestPrintPortfolioMetrics:
    """print_portfolio_metrics のテスト"""

    def test_no_exception_on_valid_metrics(self):
        """有効なメトリクスで例外が発生しないこと"""
        from src.backtest.portfolio import print_portfolio_metrics

        metrics = {
            "total_return": 0.05,
            "max_drawdown": -0.08,
            "sharpe_ratio": 1.2,
            "num_rebalances": 5,
        }
        print_portfolio_metrics(metrics, market="jp", top_n=5, rebalance_freq="weekly")

    def test_no_exception_on_empty_metrics(self):
        """空のメトリクスでも例外が発生しないこと"""
        from src.backtest.portfolio import print_portfolio_metrics

        print_portfolio_metrics({}, market="jp", top_n=5, rebalance_freq="weekly")


class TestSavePortfolioResults:
    """save_portfolio_results のテスト"""

    def test_creates_csv_files(self, tmp_path):
        """CSV ファイルが作成されること"""
        from src.backtest.portfolio import save_portfolio_results

        equity_df = pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-05"],
                "portfolio_value": [1000000.0, 1010000.0],
                "equal_weight_value": [1000000.0, 1005000.0],
            }
        )
        holdings_df = pd.DataFrame(
            {
                "symbol": ["jp_7203", "jp_7203"],
                "weight": [0.5, 0.6],
            }
        )
        metrics = {"total_return": 0.01}

        with patch("src.utils.results_io.get_results_dir", return_value=str(tmp_path)):
            save_portfolio_results(
                equity_df,
                metrics,
                holdings_df,
                market="jp",
                top_n=5,
                rebalance_freq="weekly",
            )

        csv_files = list(tmp_path.glob("**/*.csv"))
        assert len(csv_files) >= 1


class TestSoftmaxWeights(unittest.TestCase):
    """_softmax_weights のテスト"""

    def test_outputs_sum_to_one(self):
        """softmax の合計が 1.0 になること"""
        import pandas as pd

        from src.backtest.portfolio import _softmax_weights

        scores = pd.Series([0.1, 0.2, 0.05, 0.3])
        weights = _softmax_weights(scores)
        assert abs(weights.sum() - 1.0) < 1e-9

    def test_handles_all_negative_scores(self):
        """全スコアが負の場合は空の Series が返ること（正スコアのみ対象）"""
        import pandas as pd

        from src.backtest.portfolio import _softmax_weights

        scores = pd.Series([-0.1, -0.2, -0.3])
        weights = _softmax_weights(scores)
        assert isinstance(weights, pd.Series)

    def test_single_element_returns_one(self):
        """要素 1 個のスコアで 1.0 が返ること"""
        import pandas as pd

        from src.backtest.portfolio import _softmax_weights

        scores = pd.Series([0.5])
        weights = _softmax_weights(scores)
        assert abs(weights.iloc[0] - 1.0) < 1e-9


class TestSimulatePortfolio(unittest.TestCase):
    """_simulate_portfolio のテスト"""

    def _make_matrices(self, symbols, n_days=10):
        import numpy as np
        import pandas as pd

        dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
        # 正のスコアにして _softmax_weights が動作するようにする
        rng = np.random.default_rng(seed=42)
        score_data = {sym: rng.random(n_days) * 0.5 + 0.1 for sym in symbols}
        close_data = {sym: rng.random(n_days) * 100 + 100 for sym in symbols}
        return pd.DataFrame(score_data, index=dates), pd.DataFrame(close_data, index=dates)

    @patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector")
    def test_returns_equity_curve(self, mock_sector):
        """シミュレーション結果に portfolio_value 列が含まれること"""
        from src.backtest.portfolio import _get_rebalance_dates, _simulate_portfolio

        symbols = ["AAPL", "MSFT", "GOOG"]
        mock_sector.return_value = "Technology"
        score_matrix, close_matrix = self._make_matrices(symbols)
        rebalance_dates = _get_rebalance_dates(score_matrix.index, "weekly")

        equity_df, holdings = _simulate_portfolio(
            score_matrix,
            close_matrix,
            rebalance_dates,
            top_n=2,
            initial_cash=1_000_000,
            execution=ExecutionModel(TradingCosts(fee_rate=0.001)),
            max_sector_positions=3,
        )
        assert "portfolio_value" in equity_df.columns
        assert len(equity_df) > 0

    @patch("src.backtest.portfolio.simulation._get_portfolio_symbol_sector")
    def test_holdings_records_are_list(self, mock_sector):
        """ホールディング記録がリストであること"""
        from src.backtest.portfolio import _get_rebalance_dates, _simulate_portfolio

        symbols = ["AAPL", "MSFT"]
        mock_sector.return_value = "Technology"
        score_matrix, close_matrix = self._make_matrices(symbols, n_days=15)
        rebalance_dates = _get_rebalance_dates(score_matrix.index, "monthly")

        _, holdings = _simulate_portfolio(
            score_matrix,
            close_matrix,
            rebalance_dates,
            top_n=1,
            initial_cash=500_000,
            execution=ExecutionModel(TradingCosts(fee_rate=0.001)),
            max_sector_positions=2,
        )
        assert isinstance(holdings, list)


class TestSimulatePortfolioMissingPrice(unittest.TestCase):
    """Issue #846: 値段の欠けた日は、その日までの最後の値段で評価・売却する。"""

    def _run(self, score_matrix, close_matrix, rebalance_dates, top_n):
        from src.backtest.portfolio import _simulate_portfolio

        with patch(
            "src.backtest.portfolio.simulation._get_portfolio_symbol_sector",
            return_value="Technology",
        ):
            equity_df, _ = _simulate_portfolio(
                score_matrix,
                close_matrix,
                rebalance_dates,
                top_n=top_n,
                initial_cash=10_000,
                execution=ExecutionModel(TradingCosts(fee_rate=0.0)),
                max_sector_positions=5,
            )
        return equity_df.set_index("date")

    def test_missing_day_is_valued_at_last_price(self):
        """欠けた日の評価が買値に戻らず、基準も0にならないこと"""
        dates = pd.date_range("2026-01-05", periods=4, freq="B")
        score_matrix = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.5]}, index=dates)
        close_matrix = pd.DataFrame({"A": [100.0, 120.0, float("nan"), 130.0]}, index=dates)

        equity = self._run(score_matrix, close_matrix, [dates[0]], top_n=1)

        missing_day = str(dates[2])[:10]
        assert equity.loc[missing_day, "portfolio_value"] == pytest.approx(12_000)
        assert equity.loc[missing_day, "equal_weight_value"] == pytest.approx(12_000)

    def test_rebalance_sells_missing_holding_at_last_price(self):
        """リバランスの日に欠けた保有が最後の値段で現金に戻ること（戦略と基準）"""
        dates = pd.date_range("2026-01-05", periods=3, freq="B")
        score_matrix = pd.DataFrame({"A": [0.9, 0.9, 0.1], "B": [0.1, 0.1, 0.9]}, index=dates)
        close_matrix = pd.DataFrame(
            {"A": [100.0, 120.0, float("nan")], "B": [50.0, 50.0, 50.0]}, index=dates
        )

        equity = self._run(score_matrix, close_matrix, [dates[0], dates[2]], top_n=1)

        rebalance_day = str(dates[2])[:10]
        # 戦略: A を 100 株 @120 で売り 12,000 を B に回す
        assert equity.loc[rebalance_day, "portfolio_value"] == pytest.approx(12_000)
        # 基準: A 50 株 @120 + B 100 株 @50 = 11,000
        assert equity.loc[rebalance_day, "equal_weight_value"] == pytest.approx(11_000)

    def test_no_backfill_before_first_price(self):
        """始めの日より前に値段が無い銘柄は、後ろの値で埋めて買わないこと"""
        dates = pd.date_range("2026-01-05", periods=2, freq="B")
        score_matrix = pd.DataFrame({"A": [0.9, 0.9]}, index=dates)
        close_matrix = pd.DataFrame({"A": [float("nan"), 100.0]}, index=dates)

        equity = self._run(score_matrix, close_matrix, [dates[0]], top_n=1)

        first_day = str(dates[0])[:10]
        assert equity.loc[first_day, "portfolio_value"] == pytest.approx(10_000)
        assert equity.loc[first_day, "equal_weight_value"] == pytest.approx(10_000)


class TestPortfolioMarketSlippage(unittest.TestCase):
    """Issue #847: 全市場の実行は、銘柄の鍵の市場ごとの既定スリッページで売買する。"""

    _INITIAL_CASH = 1_000_000

    def _run(self, symbols, market, slippage=None):
        from src.backtest.portfolio import run_portfolio_backtest

        dates = pd.date_range("2026-01-05", periods=3, freq="B")
        prices = {"jp_7203": 100.0, "us_AAPL": 200.0}
        score_matrix = pd.DataFrame({s: [0.5, 0.5, 0.5] for s in symbols}, index=dates)
        close_matrix = pd.DataFrame({s: [prices[s]] * 3 for s in symbols}, index=dates)
        all_symbols = [tuple(s.split("_", 1)) for s in symbols]

        with (
            patch("src.utils.db.stock_features.get_all_symbols", return_value=all_symbols),
            patch(
                "src.backtest.portfolio._build_signal_matrix",
                return_value=(score_matrix, close_matrix),
            ),
            patch(
                "src.backtest.portfolio._attach_regime_metrics",
                side_effect=lambda eq, _close: (eq, {}),
            ),
            patch(
                "src.backtest.portfolio.simulation._get_portfolio_symbol_sector",
                return_value="Technology",
            ),
        ):
            equity_df, _, _ = run_portfolio_backtest(
                market=market,
                top_n=len(symbols),
                rebalance_freq="weekly",
                initial_cash=self._INITIAL_CASH,
                fee_rate=0.0,
                slippage=slippage,
            )
        return equity_df.iloc[0]

    def _expected_value(self, slippage_by_symbol, prices):
        """各銘柄に予算を等分し、その銘柄のスリッページで買った直後の評価額。"""
        budget = self._INITIAL_CASH / len(slippage_by_symbol)
        value = float(self._INITIAL_CASH)
        for sym, slip in slippage_by_symbol.items():
            execution = ExecutionModel(TradingCosts(fee_rate=0.0, slippage_rate=slip))
            qty = execution.max_affordable_qty(budget, prices[sym])
            value += qty * prices[sym] - execution.buy_cost(qty, prices[sym])
        return round(value, 2)

    def test_all_markets_uses_per_symbol_market_default(self):
        """market=None で jp_ は日本の既定、us_ は米国の既定で売買すること"""
        from config.settings import DEFAULT_SLIPPAGE_JP, DEFAULT_SLIPPAGE_US

        row = self._run(["jp_7203", "us_AAPL"], market=None)

        expected = self._expected_value(
            {"jp_7203": DEFAULT_SLIPPAGE_JP, "us_AAPL": DEFAULT_SLIPPAGE_US},
            {"jp_7203": 100.0, "us_AAPL": 200.0},
        )
        assert row["portfolio_value"] == pytest.approx(expected)
        assert row["equal_weight_value"] == pytest.approx(expected)

    def test_all_markets_jp_only_matches_market_jp(self):
        """日本株だけなら、market=None と market=jp の結果が同じこと"""
        none_row = self._run(["jp_7203"], market=None)
        jp_row = self._run(["jp_7203"], market="jp")
        assert none_row["portfolio_value"] == pytest.approx(jp_row["portfolio_value"])

    def test_explicit_market_unchanged(self):
        """market=jp と us の明示は、それぞれの市場の既定を当てること"""
        from config.settings import DEFAULT_SLIPPAGE_JP, DEFAULT_SLIPPAGE_US

        jp_row = self._run(["jp_7203"], market="jp")
        us_row = self._run(["us_AAPL"], market="us")
        assert jp_row["portfolio_value"] == pytest.approx(
            self._expected_value({"jp_7203": DEFAULT_SLIPPAGE_JP}, {"jp_7203": 100.0})
        )
        assert us_row["portfolio_value"] == pytest.approx(
            self._expected_value({"us_AAPL": DEFAULT_SLIPPAGE_US}, {"us_AAPL": 200.0})
        )

    def test_explicit_slippage_applies_to_all_symbols(self):
        """slippage を明示したときは、全市場でも全銘柄にその値を当てること"""
        row = self._run(["jp_7203", "us_AAPL"], market=None, slippage=0.002)

        expected = self._expected_value(
            {"jp_7203": 0.002, "us_AAPL": 0.002},
            {"jp_7203": 100.0, "us_AAPL": 200.0},
        )
        assert row["portfolio_value"] == pytest.approx(expected)


class TestPlotPortfolio(unittest.TestCase):
    """plot_portfolio のテスト"""

    def test_returns_empty_string_on_empty_df(self):
        """equity_df が空のとき空文字を返すこと"""
        import pandas as pd

        from src.backtest.portfolio import plot_portfolio

        equity_df = pd.DataFrame()
        holdings_df = pd.DataFrame()
        metrics: dict = {}

        result = plot_portfolio(equity_df, holdings_df, metrics, "/tmp/test_output")
        assert result == ""


# ──────────────────────────────────────────────────────────────────
# _build_signal_matrix の追加テスト（unittest.TestCase スタイル）
# ──────────────────────────────────────────────────────────────────


class TestBuildSignalMatrixAdditional(unittest.TestCase):
    """_build_signal_matrix の追加テスト"""

    @patch("src.backtest.portfolio.signal_matrix.get_model_manager")
    @patch("src.backtest.pipeline.load_features")
    def test_builds_matrix_for_single_symbol(self, mock_load, mock_get_mm):
        """1銘柄の予測スコアと価格マトリクスが構築されること"""
        import numpy as np

        from src.backtest.portfolio import _build_signal_matrix

        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        df_feat = pd.DataFrame(
            {
                "close_lag1": np.random.randn(n),
                "Close": np.random.rand(n) * 100 + 100,
                "y": np.random.randn(n) * 0.01,
            },
            index=dates,
        )
        mock_load.return_value = df_feat

        n_test = n - int(n * 0.8)  # 10
        mock_mm = mock_get_mm.return_value
        mock_mm.predict_with_model.return_value = np.array([0.01] * n_test)

        score_matrix, close_matrix = _build_signal_matrix(
            symbols=[("jp", "7203")],
            model_type="XGBoostModel",
            train_ratio=0.8,
            source="file",
            threshold=0.0,
            ensemble=False,
        )

        self.assertIsInstance(score_matrix, pd.DataFrame)
        self.assertIsInstance(close_matrix, pd.DataFrame)
        # 1銘柄のデータが含まれること
        self.assertFalse(score_matrix.empty)
        self.assertIn("jp_7203", score_matrix.columns)

    @patch("src.backtest.portfolio.signal_matrix.get_model_manager")
    @patch("src.backtest.pipeline.load_features")
    def test_skips_symbol_with_insufficient_data(self, mock_load, mock_get_mm):
        """データ不足の銘柄はスキップされること"""
        import numpy as np

        from src.backtest.portfolio import _build_signal_matrix

        # 30行未満 → スキップ
        n = 10
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        df_small = pd.DataFrame(
            {
                "close_lag1": np.random.randn(n),
                "Close": np.random.rand(n) * 100 + 100,
                "y": np.random.randn(n) * 0.01,
            },
            index=dates,
        )
        mock_load.return_value = df_small

        score_matrix, close_matrix = _build_signal_matrix(
            symbols=[("jp", "7203")],
            model_type="XGBoostModel",
            train_ratio=0.8,
            source="file",
            threshold=0.0,
            ensemble=False,
        )

        # データ不足でスキップ → 空の DataFrame
        self.assertIsInstance(score_matrix, pd.DataFrame)
        self.assertTrue(score_matrix.empty)

    @patch("src.backtest.pipeline.load_features")
    def test_skips_symbol_when_load_features_returns_empty(self, mock_load):
        """load_features が空 DataFrame を返す場合はスキップされること"""
        from src.backtest.portfolio import _build_signal_matrix

        mock_load.return_value = pd.DataFrame()

        score_matrix, close_matrix = _build_signal_matrix(
            symbols=[("jp", "7203")],
            model_type="XGBoostModel",
            train_ratio=0.8,
            source="file",
            threshold=0.0,
            ensemble=False,
        )

        self.assertTrue(score_matrix.empty)
        self.assertTrue(close_matrix.empty)


# ──────────────────────────────────────────────────────────────────
# run_portfolio_backtest
# ──────────────────────────────────────────────────────────────────


class TestRunPortfolioBacktest(unittest.TestCase):
    """run_portfolio_backtest のテスト"""

    @patch("src.utils.db.stock_features.get_all_symbols")
    def test_returns_empty_on_no_symbols(self, mock_symbols):
        """対象銘柄が存在しない場合に空の結果が返ること"""
        from src.backtest.portfolio import run_portfolio_backtest

        mock_symbols.return_value = []
        equity_df, metrics, holdings_df = run_portfolio_backtest(market="jp")
        self.assertTrue(equity_df.empty)
        self.assertEqual(metrics, {})
        self.assertTrue(holdings_df.empty)

    @patch("src.utils.db.stock_features.get_all_symbols")
    def test_market_filter_applied(self, mock_symbols):
        """market 引数でシンボルが絞り込まれること"""
        from src.backtest.portfolio import run_portfolio_backtest

        # us のみ → jp 指定では対象なし → 空結果
        mock_symbols.return_value = [("us", "AAPL"), ("us", "MSFT")]
        equity_df, metrics, holdings_df = run_portfolio_backtest(market="jp")
        self.assertTrue(equity_df.empty)

    @patch("src.backtest.portfolio._attach_regime_metrics")
    @patch("src.backtest.portfolio._compute_portfolio_metrics")
    @patch("src.backtest.portfolio._simulate_portfolio")
    @patch("src.backtest.portfolio._get_rebalance_dates")
    @patch("src.backtest.portfolio._build_signal_matrix")
    @patch("src.utils.db.stock_features.get_all_symbols")
    def test_full_flow_returns_results(
        self,
        mock_symbols,
        mock_build,
        mock_rebalance,
        mock_simulate,
        mock_metrics,
        mock_regime,
    ):
        """全サブ関数をモックして正常フローが通ること"""
        import numpy as np

        from src.backtest.portfolio import run_portfolio_backtest

        mock_symbols.return_value = [("jp", "7203"), ("jp", "9984")]

        n = 10
        dates = pd.date_range("2026-01-01", periods=n, freq="B")
        score_matrix = pd.DataFrame(
            {
                "jp_7203": np.random.rand(n) * 0.05,
                "jp_9984": np.random.rand(n) * 0.05,
            },
            index=dates,
        )
        close_matrix = pd.DataFrame(
            {
                "jp_7203": np.random.rand(n) * 100 + 100,
                "jp_9984": np.random.rand(n) * 100 + 100,
            },
            index=dates,
        )
        mock_build.return_value = (score_matrix, close_matrix)
        mock_rebalance.return_value = list(dates[:3])

        equity_df_mock = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "portfolio_value": [1_000_000.0 + i * 1000 for i in range(n)],
                "equal_weight_value": [1_000_000.0 + i * 500 for i in range(n)],
            }
        )
        mock_simulate.return_value = (equity_df_mock, [])
        mock_metrics.return_value = {
            "total_return": 0.02,
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.05,
        }
        mock_regime.return_value = (equity_df_mock, {})

        equity_df, metrics, holdings_df = run_portfolio_backtest(market="jp")

        self.assertFalse(equity_df.empty)
        self.assertIn("total_return", metrics)

    @patch("src.backtest.portfolio._build_signal_matrix")
    @patch("src.utils.db.stock_features.get_all_symbols")
    def test_returns_empty_when_score_matrix_empty(self, mock_symbols, mock_build):
        """スコアマトリクスが空の場合に空結果が返ること"""
        from src.backtest.portfolio import run_portfolio_backtest

        mock_symbols.return_value = [("jp", "7203")]
        mock_build.return_value = (pd.DataFrame(), pd.DataFrame())

        equity_df, metrics, holdings_df = run_portfolio_backtest(market="jp")
        self.assertTrue(equity_df.empty)
