"""
バックテスト拡張機能のユニットテスト

ストップロス / テイクプロフィット / ポジションサイジング / アンサンブル予測を
外部依存なしでテストする。
"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from src.backtest.backtester import Backtester
from src.backtest.task import ReturnRegressionTask


def _make_backtester(
    initial_cash=1_000_000,
    fee_rate=0.0,
    slippage=0.0,
    stop_loss_pct=None,
    take_profit_pct=None,
    position_sizing="full",
    position_fraction=0.5,
) -> Backtester:
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
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_sizing=position_sizing,
        position_fraction=position_fraction,
    )


def _price_df(prices: list, col="Close") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({col: prices}, index=idx)


def _signal(values: list, df: pd.DataFrame) -> pd.Series:
    return pd.Series(values, index=df.index)


def _pred(values: list, df: pd.DataFrame) -> pd.Series:
    return pd.Series(values, index=df.index)


# ===========================================================================
# ストップロス
# ===========================================================================
class TestStopLoss(unittest.TestCase):

    def test_stop_loss_triggers_on_drop(self):
        """株価が5%下落したらストップロスが発動する"""
        bt = _make_backtester(initial_cash=100_000, stop_loss_pct=0.05)
        # 100円で買い → 94円に下落（6%下落 > 5%閾値）
        df = _price_df([100.0, 100.0, 94.0, 90.0])
        sig = _signal([1, 0, 0, 0], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        actions = result_df["action"].tolist()
        self.assertIn("stop_loss", actions)

    def test_stop_loss_not_triggered_within_range(self):
        """下落率が閾値内ならストップロスは発動しない"""
        bt = _make_backtester(initial_cash=100_000, stop_loss_pct=0.10)
        # 100円で買い → 95円に下落（5%下落 < 10%閾値）→ 売りシグナルで決済
        df = _price_df([100.0, 100.0, 95.0, 95.0])
        sig = _signal([1, 0, 0, -1], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        actions = result_df["action"].tolist()
        self.assertNotIn("stop_loss", actions)
        self.assertIn("sell", actions)

    def test_stop_loss_reduces_max_loss(self):
        """ストップロスありの最大損失 < なしの最大損失"""
        df = _price_df([100.0, 100.0, 80.0, 60.0])
        sig = _signal([1, 0, 0, -1], df)

        bt_no_sl = _make_backtester(initial_cash=100_000)
        _, metrics_no_sl = bt_no_sl.simulate_trading(df, sig)

        bt_sl = _make_backtester(initial_cash=100_000, stop_loss_pct=0.10)
        _, metrics_sl = bt_sl.simulate_trading(df, sig)

        # ストップロスありの方が損失が小さい（リターンが高い）
        self.assertGreater(metrics_sl["total_return"], metrics_no_sl["total_return"])


# ===========================================================================
# テイクプロフィット
# ===========================================================================
class TestTakeProfit(unittest.TestCase):

    def test_take_profit_triggers_on_gain(self):
        """株価が10%上昇したらテイクプロフィットが発動する"""
        bt = _make_backtester(initial_cash=100_000, take_profit_pct=0.10)
        # 100円で買い → 111円に上昇（11%上昇 > 10%閾値）
        df = _price_df([100.0, 100.0, 111.0, 120.0])
        sig = _signal([1, 0, 0, 0], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        actions = result_df["action"].tolist()
        self.assertIn("take_profit", actions)

    def test_take_profit_not_triggered_within_range(self):
        """上昇率が閾値内ならテイクプロフィットは発動しない"""
        bt = _make_backtester(initial_cash=100_000, take_profit_pct=0.20)
        df = _price_df([100.0, 100.0, 110.0, 110.0])
        sig = _signal([1, 0, 0, -1], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        actions = result_df["action"].tolist()
        self.assertNotIn("take_profit", actions)
        self.assertIn("sell", actions)

    def test_take_profit_locks_in_gains(self):
        """テイクプロフィットで利益を確定できる（その後下落しても保護）"""
        bt = _make_backtester(initial_cash=100_000, take_profit_pct=0.10)
        # 100円で買い → 115円で利確 → 80円に下落（ポジションなし）
        df = _price_df([100.0, 100.0, 115.0, 80.0])
        sig = _signal([1, 0, 0, 0], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        self.assertGreater(metrics["total_return"], 0.0)


# ===========================================================================
# ストップロス + テイクプロフィット 併用
# ===========================================================================
class TestStopLossAndTakeProfit(unittest.TestCase):

    def test_stop_loss_priority_over_buy_signal(self):
        """ストップロス発動日に買いシグナルが出ても、まず損切りが優先される"""
        bt = _make_backtester(initial_cash=100_000, stop_loss_pct=0.05)
        # 100円で買い → 93円に下落（7%下落）→ 同日買いシグナルあり
        df = _price_df([100.0, 100.0, 93.0])
        sig = _signal([1, 0, 1], df)  # 3日目は buy シグナルだが SL 優先
        result_df, metrics = bt.simulate_trading(df, sig)
        actions = result_df["action"].tolist()
        self.assertIn("stop_loss", actions)


# ===========================================================================
# ポジションサイジング
# ===========================================================================
class TestPositionSizing(unittest.TestCase):

    def test_full_sizing_uses_all_cash(self):
        """full モードでは全額使って購入する"""
        bt = _make_backtester(initial_cash=100_000, position_sizing="full")
        df = _price_df([100.0, 100.0, 120.0])
        sig = _signal([1, 0, -1], df)
        result_df, _ = bt.simulate_trading(df, sig)
        buy_row = result_df[result_df["action"] == "buy"].iloc[0]
        self.assertEqual(buy_row["qty"], 1000)  # 100000 / 100 = 1000

    def test_fixed_sizing_uses_fraction(self):
        """fixed モードでは指定比率分だけ購入する"""
        bt = _make_backtester(
            initial_cash=100_000, position_sizing="fixed", position_fraction=0.5,
        )
        df = _price_df([100.0, 100.0, 120.0])
        sig = _signal([1, 0, -1], df)
        result_df, _ = bt.simulate_trading(df, sig)
        buy_row = result_df[result_df["action"] == "buy"].iloc[0]
        self.assertEqual(buy_row["qty"], 500)  # 50000 / 100 = 500

    def test_confidence_sizing_scales_with_prediction(self):
        """confidence モードでは予測値が大きいほど多く購入する"""
        bt_low = _make_backtester(initial_cash=100_000, position_sizing="confidence")
        bt_high = _make_backtester(initial_cash=100_000, position_sizing="confidence")
        df = _price_df([100.0, 100.0, 120.0])
        sig = _signal([1, 0, -1], df)

        pred_low = _pred([0.001, 0.0, -0.001], df)   # 弱い確信
        pred_high = _pred([0.05, 0.0, -0.001], df)    # 強い確信

        result_low, _ = bt_low.simulate_trading(df, sig, pred=pred_low)
        result_high, _ = bt_high.simulate_trading(df, sig, pred=pred_high)

        qty_low = result_low[result_low["action"] == "buy"].iloc[0]["qty"]
        qty_high = result_high[result_high["action"] == "buy"].iloc[0]["qty"]
        self.assertGreater(qty_high, qty_low)

    def test_confidence_without_pred_falls_back_to_full(self):
        """confidence モードで pred=None の場合は full と同じ動作になる"""
        bt = _make_backtester(initial_cash=100_000, position_sizing="confidence")
        df = _price_df([100.0, 100.0, 120.0])
        sig = _signal([1, 0, -1], df)
        result_df, _ = bt.simulate_trading(df, sig, pred=None)
        buy_row = result_df[result_df["action"] == "buy"].iloc[0]
        self.assertEqual(buy_row["qty"], 1000)


# ===========================================================================
# メトリクスとの統合
# ===========================================================================
class TestMetricsWithNewActions(unittest.TestCase):
    """stop_loss/take_profit がメトリクスに正しく反映される"""

    def test_stop_loss_counted_as_trade(self):
        bt = _make_backtester(initial_cash=100_000, stop_loss_pct=0.05)
        df = _price_df([100.0, 100.0, 90.0, 85.0])
        sig = _signal([1, 0, 0, 0], df)
        _, metrics = bt.simulate_trading(df, sig)
        self.assertEqual(metrics["num_trades"], 1)
        self.assertAlmostEqual(metrics["win_rate"], 0.0)

    def test_take_profit_counted_as_win(self):
        bt = _make_backtester(initial_cash=100_000, take_profit_pct=0.10)
        df = _price_df([100.0, 100.0, 115.0, 120.0])
        sig = _signal([1, 0, 0, 0], df)
        _, metrics = bt.simulate_trading(df, sig)
        self.assertEqual(metrics["num_trades"], 1)
        self.assertAlmostEqual(metrics["win_rate"], 1.0)


# ===========================================================================
# 後方互換性
# ===========================================================================
class TestBackwardCompatibility(unittest.TestCase):
    """新パラメータなしで既存動作と同一であることを確認"""

    def test_default_params_match_original_behavior(self):
        """デフォルトパラメータでは従来と同じ全額投入・SL/TPなし"""
        bt = _make_backtester(initial_cash=100_000)
        self.assertIsNone(bt.stop_loss_pct)
        self.assertIsNone(bt.take_profit_pct)
        self.assertEqual(bt.position_sizing, "full")

    def test_simulate_trading_without_pred(self):
        """pred なしの呼び出しが正常動作する（後方互換）"""
        bt = _make_backtester(initial_cash=100_000)
        df = _price_df([100.0, 100.0, 120.0])
        sig = _signal([1, 0, -1], df)
        result_df, metrics = bt.simulate_trading(df, sig)
        self.assertEqual(metrics["num_trades"], 1)
        self.assertGreater(metrics["total_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
