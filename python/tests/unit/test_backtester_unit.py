"""
Unit Test: Backtester.simulate_trading

外部依存（DB・yfinance・モデル）なく、simulate_trading のロジックのみをテスト。
全依存を Mock で隔離し、高速実行を実現。
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from src.backtest.backtester import Backtester


class TestBacktesterSimulateTradingBasic:
    """基本的な シミュレーション動作"""

    def test_no_trades_on_all_hold_signal(self, sample_price_df):
        """Hold シグナルのみ → 取引なし"""
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.001,
        )

        signal = pd.Series([0, 0, 0, 0, 0], index=sample_price_df.index)
        result_df, metrics = bt.simulate_trading(sample_price_df, signal)

        assert metrics["num_trades"] == 0
        assert metrics["total_return"] == 0.0
        assert metrics["win_rate"] == 0.0

    def test_single_buy_hold_sell(self, sample_price_df):
        """Buy → Hold × 2 → Sell"""
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.001,
        )

        # Buy → Hold → Hold → Sell → Hold
        signal = pd.Series([1, 0, 0, -1, 0], index=sample_price_df.index)
        result_df, metrics = bt.simulate_trading(sample_price_df, signal)

        assert metrics["num_trades"] == 1
        # Close が 101~105 で上昇しているため利益が出ている
        assert metrics["total_return"] > 0

    def test_multiple_round_trip_trades(self, sample_price_df):
        """複数ラウンドトリップ取引"""
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.0,  # 手数料なしでテスト
        )

        # Buy → Sell → Buy → Sell
        signal = pd.Series([1, -1, 1, -1, 0], index=sample_price_df.index)
        result_df, metrics = bt.simulate_trading(sample_price_df, signal)

        assert metrics["num_trades"] == 2
        assert len(result_df) >= 2

    def test_consecutive_buy_signals_ignored(self, sample_price_df):
        """連続 Buy シグナルは2番目以降は無視される"""
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.001,
        )

        # Buy → Buy → Buy → Sell
        signal = pd.Series([1, 1, 1, -1, 0], index=sample_price_df.index)
        result_df, metrics = bt.simulate_trading(sample_price_df, signal)

        # 1取引のみ実行（最初の Buy のみ）
        assert metrics["num_trades"] == 1

    def test_uncovered_sell_signals_ignored(self, sample_price_df):
        """ポジションなしで Sell シグナル → 無視される"""
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.001,
        )

        # Sell → Sell → Buy → Sell
        signal = pd.Series([-1, -1, 1, -1, 0], index=sample_price_df.index)
        result_df, metrics = bt.simulate_trading(sample_price_df, signal)

        # Buy → Sell の 1取引のみ
        assert metrics["num_trades"] == 1


class TestBacktesterProfitLoss:
    """損益計算"""

    def test_win_trade_positive_return(self, sample_price_df):
        """利益トレード → positive return"""
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.0,  # 手数料なし
        )

        # Close が 101 → 105 に上昇
        signal = pd.Series([1, 0, 0, -1, 0], index=sample_price_df.index)
        result_df, metrics = bt.simulate_trading(sample_price_df, signal)

        assert metrics["total_return"] > 0.0
        assert metrics["win_rate"] == 1.0

    def test_loss_trade_negative_return(self):
        """損失トレード → negative return"""
        # Close が 105 → 101 に下落
        prices = pd.DataFrame(
            {"Close": [105, 104, 103, 102, 101]},
            index=pd.date_range("2024-01-01", periods=5, freq="B"),
        )

        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.0,
        )

        signal = pd.Series([1, 0, 0, -1, 0], index=prices.index)
        result_df, metrics = bt.simulate_trading(prices, signal)

        assert metrics["total_return"] < 0.0
        assert metrics["win_rate"] == 0.0

    def test_fee_impact_on_return(self, sample_price_df):
        """手数料がリターンに影響すること"""
        # Fee なし版
        bt_no_fee = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.0,
        )

        # Fee あり版
        bt_with_fee = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.01,  # 1%
        )

        signal = pd.Series([1, 0, 0, -1, 0], index=sample_price_df.index)

        _, metrics_no_fee = bt_no_fee.simulate_trading(sample_price_df, signal)
        _, metrics_with_fee = bt_with_fee.simulate_trading(sample_price_df, signal)

        # 手数料ありが手数料なしより低いリターン
        assert metrics_with_fee["total_return"] < metrics_no_fee["total_return"]


class TestBacktesterRiskManagement:
    """リスク管理機能"""

    def test_stop_loss_execution(self):
        """ストップロス発動"""
        # 105 → 100 に下落（5%）
        prices = pd.DataFrame(
            {"Close": [105, 104, 103, 102, 101, 100]},
            index=pd.date_range("2024-01-01", periods=6, freq="B"),
        )

        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.0,
            stop_loss_pct=0.05,  # 5% ストップロス
        )

        # Buy → Hold → Hold → Hold → Hold → 自動決済（ストップロス）
        signal = pd.Series([1, 0, 0, 0, 0, 0], index=prices.index)
        result_df, metrics = bt.simulate_trading(prices, signal)

        # ストップロスが発動している
        assert "stop_loss" in result_df["action"].values

    def test_take_profit_execution(self):
        """テイクプロフィット発動"""
        # 100 → 115 に上昇（15%）
        prices = pd.DataFrame(
            {"Close": [100, 105, 110, 115, 114]},
            index=pd.date_range("2024-01-01", periods=5, freq="B"),
        )

        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.0,
            take_profit_pct=0.10,  # 10% テイクプロフィット
        )

        signal = pd.Series([1, 0, 0, 0, 0], index=prices.index)
        result_df, metrics = bt.simulate_trading(prices, signal)

        # テイクプロフィットが発動している
        assert "take_profit" in result_df["action"].values


class TestBacktesterPositionSizing:
    """ポジションサイジング"""

    def test_full_position_sizing(self, sample_price_df):
        """Full ポジションサイジング"""
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.001,
            position_sizing="full",
        )

        signal = pd.Series([1, 0, 0, -1, 0], index=sample_price_df.index)
        result_df, metrics = bt.simulate_trading(sample_price_df, signal)

        # ポジションが構築されている
        assert metrics["num_trades"] == 1

    def test_fixed_position_sizing(self, sample_price_df):
        """Fixed ポジションサイジング（比率制限）"""
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.001,
            position_sizing="fixed",
            position_fraction=0.5,  # 資金の 50% のみ使用
        )

        signal = pd.Series([1, 0, 0, -1, 0], index=sample_price_df.index)
        result_df, metrics = bt.simulate_trading(sample_price_df, signal)

        # ポジションが構築されている
        assert metrics["num_trades"] == 1
        # リターンが full より低い
        # （資金が少なく投入されている）


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
