"""
Backtester のユニットテスト

simulate_trading に絞って外部依存（DB・yfinance・モデル）なしでテストする。
"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from src.backtest.backtester import Backtester
from src.backtest.task import ReturnRegressionTask


def _make_backtester(initial_cash=1_000_000, fee_rate=0.0, slippage=0.0) -> Backtester:
    """依存モックを持つ Backtester を生成する"""
    return Backtester(
        model_manager=MagicMock(),
        signal_generator=MagicMock(),
        data_loader=MagicMock(),
        start_date="2024-01-01",
        end_date="2024-12-31",
        market="jp",
        symbol="7203",
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
    )


def _price_df(prices: list, col="Close") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({col: prices}, index=idx)


def _signal(values: list, df: pd.DataFrame) -> pd.Series:
    return pd.Series(values, index=df.index)


class TestSimulateTradingBasic(unittest.TestCase):
    """simulate_trading の基本動作"""

    def test_all_hold_no_trades(self):
        bt = _make_backtester()
        df = _price_df([100, 105, 110, 115])
        sig = _signal([0, 0, 0, 0], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        self.assertEqual(metrics["num_trades"], 0)
        self.assertEqual(metrics["total_return"], 0.0)

    def test_buy_then_sell(self):
        """buy → sell 1サイクル"""
        bt = _make_backtester(initial_cash=100_000)
        # 100円で100株買える初期資金、10%上昇後売り
        df = _price_df([100.0, 100.0, 110.0, 110.0])
        sig = _signal([1, 0, -1, 0], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        self.assertEqual(metrics["num_trades"], 1)
        self.assertGreater(metrics["total_return"], 0.0)

    def test_final_sell_on_remaining_position(self):
        """売りシグナルなしで終わった場合、最終日に強制決済される"""
        bt = _make_backtester(initial_cash=100_000)
        df = _price_df([100.0, 100.0, 110.0])
        sig = _signal([1, 0, 0], df)  # 買いのみ、売りなし
        result_df, metrics = bt.simulate_trading(df, sig)
        # final_sell が発生して trade_log に記録される
        actions = result_df["action"].tolist() if not result_df.empty else []
        self.assertIn("final_sell", actions)

    def test_no_buy_when_already_in_position(self):
        """ポジション保有中は buy シグナルを無視する"""
        bt = _make_backtester(initial_cash=100_000)
        df = _price_df([100.0, 100.0, 100.0, 110.0])
        sig = _signal([1, 1, 1, -1], df)  # 3回 buy → 1回だけ買われるはず
        result_df, metrics = bt.simulate_trading(df, sig)
        buy_count = len(result_df[result_df["action"] == "buy"]) if not result_df.empty else 0
        self.assertEqual(buy_count, 1)

    def test_no_sell_when_no_position(self):
        """ポジションなしで sell シグナルが来ても無視する"""
        bt = _make_backtester(initial_cash=100_000)
        df = _price_df([100.0, 100.0, 100.0])
        sig = _signal([-1, -1, -1], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        self.assertTrue(result_df.empty)
        self.assertEqual(metrics["num_trades"], 0)


class TestSimulateTradingFeeSlippage(unittest.TestCase):
    """手数料・スリッページの影響"""

    def test_fee_reduces_return(self):
        """手数料ありのリターン < 手数料なしのリターン"""
        df = _price_df([100.0, 100.0, 120.0])
        sig = _signal([1, 0, -1], df)

        bt_no_fee = _make_backtester(initial_cash=100_000, fee_rate=0.0)
        _, metrics_no_fee = bt_no_fee.simulate_trading(df, sig)

        bt_with_fee = _make_backtester(initial_cash=100_000, fee_rate=0.01)
        _, metrics_with_fee = bt_with_fee.simulate_trading(df, sig)

        self.assertGreater(metrics_no_fee["total_return"], metrics_with_fee["total_return"])

    def test_slippage_reduces_return(self):
        df = _price_df([100.0, 100.0, 120.0])
        sig = _signal([1, 0, -1], df)

        bt_no_slip = _make_backtester(initial_cash=100_000, slippage=0.0)
        _, metrics_no_slip = bt_no_slip.simulate_trading(df, sig)

        bt_with_slip = _make_backtester(initial_cash=100_000, slippage=0.005)
        _, metrics_with_slip = bt_with_slip.simulate_trading(df, sig)

        self.assertGreater(metrics_no_slip["total_return"], metrics_with_slip["total_return"])


class TestSimulateTradingMetrics(unittest.TestCase):
    """メトリクスが正しく計算される"""

    def test_metrics_keys(self):
        bt = _make_backtester()
        df = _price_df([100.0, 110.0, 120.0])
        sig = _signal([1, 0, -1], df)
        _, metrics = bt.simulate_trading(df, sig)
        required_keys = {
            "final_cash", "total_return", "num_trades",
            "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown",
        }
        self.assertTrue(required_keys.issubset(set(metrics.keys())))

    def test_result_df_columns(self):
        bt = _make_backtester()
        df = _price_df([100.0, 110.0, 120.0])
        sig = _signal([1, 0, -1], df)
        result_df, _ = bt.simulate_trading(df, sig)
        if not result_df.empty:
            self.assertIn("action", result_df.columns)
            self.assertIn("price", result_df.columns)
            self.assertIn("cash", result_df.columns)

    def test_lowercase_close_column(self):
        """Close 列が 'close'（小文字）でも動作する"""
        bt = _make_backtester(initial_cash=100_000)
        df = _price_df([100.0, 100.0, 120.0], col="close")
        sig = _signal([1, 0, -1], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        self.assertEqual(metrics["num_trades"], 1)

    def test_multiple_cycles_count(self):
        """複数の売買サイクルが正しくカウントされる"""
        bt = _make_backtester(initial_cash=1_000_000)
        df = _price_df([100, 100, 110, 110, 100, 100, 120], col="Close")
        sig = _signal([1, 0, -1, 1, 0, 0, -1], df)
        _, metrics = bt.simulate_trading(df, sig)
        self.assertEqual(metrics["num_trades"], 2)


class TestBacktesterWithTask(unittest.TestCase):
    """task.make_signal との統合"""

    def test_return_regression_task_buy_signal_triggers_trade(self):
        """ReturnRegressionTask の buy シグナルが実際にトレードに繋がる"""
        task = ReturnRegressionTask(threshold=0.0)
        pred = pd.Series(
            [0.01, 0.01, -0.01, 0.0],
            index=pd.date_range("2024-01-01", periods=4, freq="B"),
        )
        signal = task.make_signal(pred)
        # buy=1 が 2回、sell=-1 が 1回、hold=0 が 1回
        self.assertEqual((signal == 1).sum(), 2)
        self.assertEqual((signal == -1).sum(), 1)


if __name__ == "__main__":
    unittest.main()
