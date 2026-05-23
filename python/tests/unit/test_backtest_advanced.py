"""
バックテスト拡張機能のユニットテスト

ストップロス / テイクプロフィット / ポジションサイジング / アンサンブル予測を
外部依存なしでテストする。
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.backtest.backtester import Backtester


def _make_backtester(
    initial_cash=1_000_000,
    fee_rate=0.0,
    slippage=0.0,
    stop_loss_pct=None,
    take_profit_pct=None,
    position_sizing="full",
    position_fraction=0.5,
    atr_risk_pct=0.02,
    atr_multiplier=1.0,
    atr_min_fraction=0.1,
    atr_max_fraction=1.0,
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
        atr_risk_pct=atr_risk_pct,
        atr_multiplier=atr_multiplier,
        atr_min_fraction=atr_min_fraction,
        atr_max_fraction=atr_max_fraction,
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
            initial_cash=100_000,
            position_sizing="fixed",
            position_fraction=0.5,
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

        pred_low = _pred([0.001, 0.0, -0.001], df)  # 弱い確信
        pred_high = _pred([0.05, 0.0, -0.001], df)  # 強い確信

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


class TestATRPositionSizing(unittest.TestCase):
    """ATR連動ポジションサイジングのユニットテスト"""

    def _price_df_with_atr(self, prices: list, atr_value: float = 5.0) -> pd.DataFrame:
        """ATR列付きの DataFrameを作成する"""
        idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
        return pd.DataFrame({"Close": prices, "atr": atr_value}, index=idx)

    def test_atr_qty_is_smaller_than_full(self):
        """ATRモードの購入株数は full モードより小さい"""
        cash = 1_000_000
        price = 1000.0
        atr = 100.0  # ATRが価格の10%なのでリスク大きめ
        bt_atr = _make_backtester(
            initial_cash=cash, position_sizing="atr", atr_risk_pct=0.02, fee_rate=0.0
        )
        bt_full = _make_backtester(initial_cash=cash, position_sizing="full", fee_rate=0.0)
        qty_atr = bt_atr._calc_qty(cash, price, atr_value=atr)
        qty_full = bt_full._calc_qty(cash, price)
        self.assertGreater(qty_full, qty_atr)
        self.assertGreater(qty_atr, 0)

    def test_atr_qty_calculation_formula(self):
        """ATRクォンティティの計算式が正しい"""
        # risk_amount = 1_000_000 * 0.02 = 20_000
        # stop_distance = ATR(50) * multiplier(1.0) = 50
        # qty_by_risk = 20_000 / 50 = 400株
        bt = _make_backtester(
            initial_cash=1_000_000, position_sizing="atr", atr_risk_pct=0.02, atr_multiplier=1.0
        )
        qty = bt._calc_qty(1_000_000, 1000.0, atr_value=50.0)
        self.assertEqual(qty, 400)

    def test_atr_qty_capped_by_cash(self):
        """ATRモードの購入株数は現金上限でキャップされる"""
        # risk_amount が大きすぎて現金を超える場合
        # cash=100_000, price=1000, 全額引けても100株が上限
        bt = _make_backtester(
            initial_cash=100_000, position_sizing="atr", atr_risk_pct=1.0, fee_rate=0.0
        )
        qty = bt._calc_qty(100_000, 1000.0, atr_value=1.0)
        self.assertLessEqual(qty * 1000.0, 100_000)

    def test_atr_qty_respects_fraction_bounds(self):
        """ATRモードの購入株数が建玉比率の上下限に収まる"""
        bt = _make_backtester(
            initial_cash=100_000,
            position_sizing="atr",
            atr_risk_pct=0.0001,
            atr_multiplier=1.0,
            atr_min_fraction=0.1,
            atr_max_fraction=0.3,
            fee_rate=0.0,
        )
        qty = bt._calc_qty(100_000, 1000.0, atr_value=100.0)
        self.assertGreaterEqual(qty, 10)
        self.assertLessEqual(qty, 30)

    def test_atr_fallback_to_full_when_no_atr(self):
        """atr_valueがNoneの場合は full モードと同じ動作になる"""
        bt = _make_backtester(initial_cash=1_000_000, position_sizing="atr", fee_rate=0.0)
        qty_no_atr = bt._calc_qty(1_000_000, 1000.0, atr_value=None)
        bt_full = _make_backtester(initial_cash=1_000_000, position_sizing="full", fee_rate=0.0)
        qty_full = bt_full._calc_qty(1_000_000, 1000.0)
        self.assertEqual(qty_no_atr, qty_full)

    def test_atr_fallback_trade_is_counted_in_metrics(self):
        """ATR欠損で full フォールバックした回数がメトリクスに反映される"""
        bt = _make_backtester(initial_cash=100_000, position_sizing="atr", fee_rate=0.0)
        df = _price_df([100.0, 100.0, 120.0])
        sig = _signal([1, 0, -1], df)
        _, metrics = bt.simulate_trading(df, sig)
        self.assertEqual(metrics["atr_fallback_trades"], 1)
        self.assertAlmostEqual(metrics["avg_position_fraction"], 1.0)

    def test_simulate_trading_atr_mode(self):
        """ATR列付き df で atr モードのシミュレーションが正常完了する"""
        bt = _make_backtester(
            initial_cash=1_000_000, position_sizing="atr", atr_risk_pct=0.02, fee_rate=0.0
        )
        df = self._price_df_with_atr([1000.0, 1050.0, 1100.0], atr_value=50.0)
        sig = pd.Series([1, 0, -1], index=df.index)
        result_df, metrics = bt.simulate_trading(df, sig)
        self.assertEqual(metrics["num_trades"], 1)
        self.assertGreater(metrics["total_return"], 0.0)
        self.assertGreater(metrics["avg_position_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
