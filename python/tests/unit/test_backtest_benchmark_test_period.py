"""単一期間バックテストのベンチマーク期間が検証期間になることのテスト（#1062）。

戦略の total_return は検証期間全体（取引の無い期間は現金）の成績なので、
ベンチマークも取引の日付ではなく検証期間の最初と最後の日付で取る。
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest.pipeline import fetch_benchmark_for_result


def _trades_in_middle() -> pd.DataFrame:
    """検証期間（2024-01-01〜2024-06-28）の途中だけで取引する取引ログ。"""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-01", "2024-03-15"]),
            "action": ["BUY", "SELL"],
        }
    )


class TestFetchBenchmarkPeriod:
    @patch("src.backtest.metrics.fetch_benchmark_returns")
    def test_uses_test_period_when_given(self, mock_fetch):
        """検証期間が分かるときは、取引の日付ではなく検証期間で BM を取る"""
        mock_fetch.return_value = {"total_return": 0.1}

        fetch_benchmark_for_result(
            _trades_in_middle(),
            "sp500",
            "2020-01-01",
            "2024-12-31",
            test_start="2024-01-01",
            test_end="2024-06-28",
        )

        mock_fetch.assert_called_once_with("^GSPC", "2024-01-01", "2024-06-28")

    @patch("src.backtest.metrics.fetch_benchmark_returns")
    def test_falls_back_to_trade_dates_without_test_period(self, mock_fetch):
        """検証期間が分からないときは従来どおり取引の日付で BM を取る"""
        mock_fetch.return_value = {"total_return": 0.1}

        fetch_benchmark_for_result(_trades_in_middle(), "sp500", "2020-01-01", "2024-12-31")

        mock_fetch.assert_called_once_with("^GSPC", "2024-03-01", "2024-03-15")

    @patch("src.backtest.metrics.fetch_benchmark_returns")
    def test_falls_back_to_fallback_without_date_column(self, mock_fetch):
        """検証期間も date 列も無いときは fallback の日付で BM を取る"""
        mock_fetch.return_value = {"total_return": 0.1}

        fetch_benchmark_for_result(
            pd.DataFrame({"action": ["BUY"]}), "sp500", "2020-01-01", "2024-12-31"
        )

        mock_fetch.assert_called_once_with("^GSPC", "2020-01-01", "2024-12-31")


class TestRunBacktestSingleTestPeriod:
    @patch("src.backtest.pipeline.runner.load_features")
    @patch("src.backtest.pipeline.runner.get_model_manager")
    @patch("src.backtest.backtester.Backtester")
    @patch("src.trading.signal_generator.SignalGenerator")
    def test_metrics_include_test_period(self, mock_sg, mock_bt, mock_get_mm, mock_lf):
        """metrics に検証期間（test_df の最初と最後の日付）が入る"""
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        mock_lf.return_value = pd.DataFrame(
            {
                "feat1": np.random.uniform(0, 1, n),
                "y": np.random.uniform(-0.05, 0.05, n),
                "Close": [100.0 + i for i in range(n)],
            },
            index=dates,
        )
        mock_get_mm.return_value.predict_with_model.return_value = [0.01] * 4
        mock_bt.return_value.simulate_trading.return_value = (
            pd.DataFrame(),
            {"total_return": 0.01, "num_trades": 2},
        )

        from src.backtest.pipeline import run_backtest_single

        _, metrics, _ = run_backtest_single("jp", "7203")

        assert metrics["test_start"] == str(dates[16])[:10]
        assert metrics["test_end"] == str(dates[-1])[:10]
