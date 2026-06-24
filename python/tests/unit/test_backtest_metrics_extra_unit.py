"""ユニットテスト: backtest/metrics.py の追加カバレッジ"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


def _make_trade_log() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "action": ["buy", "sell", "buy", "sell", "buy", "sell"],
            "price": [1000, 1050, 1050, 1100, 1100, 1150],
            "shares": [100, 100, 100, 100, 100, 100],
            "cash": [900000, 1005000, 900000, 1010000, 900000, 1015000],
            "pnl": [0, 5000, 0, 5000, 0, 5000],
        }
    )


def _make_price_df(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(1000, 1100, n)
    return pd.DataFrame(
        {
            "Open": close - 5,
            "High": close + 10,
            "Low": close - 10,
            "Close": close,
            "Volume": np.full(n, 1_000_000, dtype=float),
        },
        index=idx,
    )


class TestComputeMetricsNoActionColumn(unittest.TestCase):
    def test_no_action_column_still_computes(self):
        from src.backtest.metrics import compute_metrics

        # action 列なしの場合
        trade_log = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="B"),
                "cash": [1_000_000, 1_010_000, 1_020_000],
                "price": [1000, 1010, 1020],
                "shares": [100, 100, 100],
                "pnl": [0, 10000, 10000],
            }
        )
        result = compute_metrics(trade_log, 1_000_000)
        self.assertIsInstance(result, dict)
        self.assertIn("total_return", result)

    def test_empty_sell_log_uses_initial_cash(self):
        from src.backtest.metrics import compute_metrics

        # action 列はあるが sell アクションなし → sell_log.empty になるパスを通る
        trade_log = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=2, freq="B"),
                "action": ["buy", "buy"],
                "cash": [900_000, 900_000],
                "price": [1000, 1000],
                "qty": [100, 100],
                "pnl": [0, 0],
            }
        )
        result = compute_metrics(trade_log, 1_000_000)
        self.assertIn("total_return", result)


class TestFetchBenchmarkReturns(unittest.TestCase):
    def test_returns_none_when_empty_df(self):
        from src.backtest.metrics import fetch_benchmark_returns

        mock_yf = MagicMock()
        mock_yf.download.return_value = pd.DataFrame()
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_benchmark_returns("^N225", "2024-01-01", "2024-12-31")
        self.assertIsNone(result["total_return"])

    def test_returns_none_when_single_row(self):
        from src.backtest.metrics import fetch_benchmark_returns

        df = pd.DataFrame(
            {"Close": [1000.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        mock_yf = MagicMock()
        mock_yf.download.return_value = df
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_benchmark_returns("^N225", "2024-01-01", "2024-12-31")
        self.assertIsNone(result["total_return"])

    def test_returns_total_return_on_valid_data(self):
        from src.backtest.metrics import fetch_benchmark_returns

        df = pd.DataFrame(
            {"Close": [1000.0, 1050.0, 1100.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )
        mock_yf = MagicMock()
        mock_yf.download.return_value = df
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_benchmark_returns("^N225", "2024-01-01", "2024-12-31")
        self.assertIsNotNone(result["total_return"])

    def test_returns_none_on_exception(self):
        from src.backtest.metrics import fetch_benchmark_returns

        mock_yf = MagicMock()
        mock_yf.download.side_effect = RuntimeError("network")
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = fetch_benchmark_returns("^N225", "2024-01-01", "2024-12-31")
        self.assertIsNone(result["total_return"])


class TestComputeMetricsByRegime(unittest.TestCase):
    def _mock_metrics(self):
        return {
            "total_return": 0.05,
            "win_rate": 0.6,
            "num_trades": 10,
            "profit_factor": 1.5,
            "sharpe_ratio": 0.7,
            "max_drawdown": -0.08,
        }

    def test_returns_all_key_when_empty_price_df(self):
        from src.backtest.metrics import compute_metrics_by_regime

        trade_log = _make_trade_log()
        with patch("src.backtest.metrics.core.compute_metrics", return_value=self._mock_metrics()):
            result = compute_metrics_by_regime(trade_log, pd.DataFrame(), 1_000_000)
        self.assertIn("all", result)
        self.assertNotIn("bull", result)

    def test_returns_all_key_when_empty_trade_log(self):
        from src.backtest.metrics import compute_metrics_by_regime

        with patch("src.backtest.metrics.core.compute_metrics", return_value=self._mock_metrics()):
            result = compute_metrics_by_regime(None, _make_price_df(), 1_000_000)
        self.assertIn("all", result)

    def test_returns_all_key_when_no_date_column(self):
        from src.backtest.metrics import compute_metrics_by_regime

        trade_log = _make_trade_log().drop(columns=["date"])
        with patch("src.backtest.metrics.core.compute_metrics", return_value=self._mock_metrics()):
            result = compute_metrics_by_regime(trade_log, _make_price_df(), 1_000_000)
        self.assertIn("all", result)

    def test_returns_regime_keys_with_valid_data(self):
        from src.backtest.metrics import compute_metrics_by_regime

        trade_log = _make_trade_log()
        price_df = _make_price_df(200)

        idx = price_df.index
        regime_series = pd.Series(["bull"] * len(idx), index=idx)

        with (
            patch("src.backtest.metrics.core.compute_metrics", return_value=self._mock_metrics()),
            patch("src.backtest.metrics.core._empty_metrics", return_value=self._mock_metrics()),
        ):
            with patch("src.market_data.technical.classify_regime", return_value=regime_series):
                result = compute_metrics_by_regime(trade_log, price_df, 1_000_000)

        self.assertIn("all", result)

    def test_empty_regime_series_returns_only_all(self):
        from src.backtest.metrics import compute_metrics_by_regime

        trade_log = _make_trade_log()
        price_df = _make_price_df()

        with patch("src.backtest.metrics.core.compute_metrics", return_value=self._mock_metrics()):
            with patch(
                "src.market_data.technical.classify_regime", return_value=pd.Series([], dtype=str)
            ):
                result = compute_metrics_by_regime(trade_log, price_df, 1_000_000)

        self.assertIn("all", result)
        self.assertNotIn("bull", result)


if __name__ == "__main__":
    unittest.main()
