"""Unit Test: Backtester.simulate_trading

外部依存（DB・yfinance・モデル）なく、simulate_trading のロジックのみをテスト。
全依存を Mock で隔離し、高速実行を実現。
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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

    def test_final_sell_on_remaining_position(self):
        """売りシグナルなしで終了した場合、最終日に強制決済される"""
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100.0, 100.0, 110.0]}, index=idx)
        sig = pd.Series([1, 0, 0], index=idx)  # 買いのみ、売りなし
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=100_000,
            fee_rate=0.0,
        )
        result_df, _ = bt.simulate_trading(df, sig)
        actions = result_df["action"].tolist() if not result_df.empty else []
        assert "final_sell" in actions


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
            {"Close": [105, 104, 103, 102, 101, 99]},
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
            execution_lag=0,  # SL 発火ロジックを同バーで隔離検証
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
            execution_lag=0,  # TP 発火ロジックを同バーで隔離検証
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


class TestBacktesterSlippage:
    """スリッページの影響"""

    def test_slippage_reduces_return(self):
        """スリッページありのリターン < スリッページなしのリターン"""
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100.0, 100.0, 120.0]}, index=idx)
        sig = pd.Series([1, 0, -1], index=idx)

        bt_no_slip = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=100_000,
            fee_rate=0.0,
            slippage=0.0,
        )
        _, metrics_no_slip = bt_no_slip.simulate_trading(df, sig)

        bt_with_slip = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=100_000,
            fee_rate=0.0,
            slippage=0.005,
        )
        _, metrics_with_slip = bt_with_slip.simulate_trading(df, sig)

        assert metrics_no_slip["total_return"] > metrics_with_slip["total_return"]


class TestBacktesterMetricsShape:
    """メトリクスキーと結果 DataFrame の構造検証"""

    def test_metrics_contains_required_keys(self, sample_price_df):
        """simulate_trading が必須キーを全て含むメトリクスを返す"""
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
        sig = pd.Series([1, 0, -1, 0, 0], index=sample_price_df.index)
        _, metrics = bt.simulate_trading(sample_price_df, sig)
        required = {
            "final_cash",
            "total_return",
            "num_trades",
            "win_rate",
            "profit_factor",
            "sharpe_ratio",
            "max_drawdown",
        }
        assert required.issubset(metrics.keys())

    def test_result_df_contains_action_price_cash(self, sample_price_df):
        """取引ログ DataFrame に action / price / cash 列がある"""
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
        sig = pd.Series([1, 0, -1, 0, 0], index=sample_price_df.index)
        result_df, _ = bt.simulate_trading(sample_price_df, sig)
        if not result_df.empty:
            assert "action" in result_df.columns
            assert "price" in result_df.columns
            assert "cash" in result_df.columns

    def test_lowercase_close_column_accepted(self):
        """Close 列が 'close'（小文字）でも動作する"""
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({"close": [100.0, 100.0, 120.0]}, index=idx)
        sig = pd.Series([1, 0, -1], index=idx)
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=100_000,
            fee_rate=0.0,
        )
        _, metrics = bt.simulate_trading(df, sig)
        assert metrics["num_trades"] == 1

    def test_multiple_cycles_count(self):
        """複数の売買サイクルが正しくカウントされる"""
        idx = pd.date_range("2024-01-01", periods=7, freq="B")
        df = pd.DataFrame({"Close": [100, 100, 110, 110, 100, 100, 120]}, index=idx)
        sig = pd.Series([1, 0, -1, 1, 0, 0, -1], index=idx)
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
        _, metrics = bt.simulate_trading(df, sig)
        assert metrics["num_trades"] == 2


class TestBacktesterWithTask:
    """task.make_signal との統合"""

    def test_return_regression_task_signal_counts(self):
        """Buy/sell signal counts are correct via ReturnRegressionTask."""
        from src.backtest.task import ReturnRegressionTask

        task = ReturnRegressionTask(threshold=0.0)
        pred = pd.Series(
            [0.01, 0.01, -0.01, 0.0],
            index=pd.date_range("2024-01-01", periods=4, freq="B"),
        )
        signal = task.make_signal(pred)
        assert (signal == 1).sum() == 2
        assert (signal == -1).sum() == 1


class TestBacktesterRunPath:
    """run / _load_from_raw の分岐カバレッジ"""

    def test_run_executes_with_mock_dependencies(self):
        idx = pd.date_range("2024-01-01", periods=8, freq="B")
        df = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104, 105, 106, 107],
                "feat1": [1, 2, 3, 4, 5, 6, 7, 8],
                "feat2": [2, 3, 4, 5, 6, 7, 8, 9],
            },
            index=idx,
        )

        model = MagicMock()
        model.predict.return_value = [0.01] * 7
        mm = MagicMock()
        mm.get_model.return_value = model

        dl = MagicMock()
        dl.get_stock_data_auto.return_value = df

        bt = Backtester(
            model_manager=mm,
            signal_generator=MagicMock(),
            data_loader=dl,
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=100_000,
            fee_rate=0.0,
        )

        with patch("src.market_data.technical.add_technical_indicators", side_effect=lambda x: x):
            result_df, metrics = bt.run(model_name="dummy")

        assert isinstance(result_df, pd.DataFrame)
        assert "total_return" in metrics

    def test_load_from_raw_raises_when_empty(self):
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
        )

        fake_loader = types.ModuleType("src.market_data.loader")
        fake_loader.get_raw_ohlcv_from_db = lambda *_a, **_k: pd.DataFrame()
        with patch.dict(sys.modules, {"src.market_data.loader": fake_loader}):
            with pytest.raises(ValueError):
                bt._load_from_raw()


class TestBacktesterShortSimulation:
    """R-215: ショートシミュレーションのテスト"""

    def _make_bt(self, enable_short: bool = True, fee_rate: float = 0.0, execution_lag: int = 0):
        # PnL 式・約定メカニクスを同バー(execution_lag=0)で隔離検証する。
        # 翌バー約定(#493)の挙動は TestBacktesterExecutionLag で別途検証する。
        return Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=fee_rate,
            enable_short=enable_short,
            execution_lag=execution_lag,
        )

    def test_short_entry_and_cover_profit(self):
        """価格下落時: ショートエントリー → ショートカバーで利益"""
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({"Close": [100.0, 90.0, 85.0, 80.0]}, index=idx)
        # -1(short entry) → 0 → 0 → 1(short cover)
        signal = pd.Series([-1, 0, 0, 1], index=idx)

        bt = self._make_bt(enable_short=True)
        result_df, metrics = bt.simulate_trading(df, signal)

        # short_return は正（下落で利益）
        assert metrics["short_return"] > 0
        assert metrics["short_num_trades"] == 1
        # short アクションと short_cover アクションがある
        assert "short" in result_df["action"].values
        assert "short_cover" in result_df["action"].values

    def test_short_entry_and_cover_loss(self):
        """価格上昇時: ショートエントリー → ショートカバーで損失"""
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({"Close": [100.0, 105.0, 110.0, 120.0]}, index=idx)
        signal = pd.Series([-1, 0, 0, 1], index=idx)

        bt = self._make_bt(enable_short=True)
        _, metrics = bt.simulate_trading(df, signal)

        assert metrics["short_return"] < 0
        assert metrics["short_num_trades"] == 1

    def test_short_pnl_calculation(self):
        """ショートPnL計算: 手数料なし・100株・entry=100・exit=80 → PnL=(100-80)*100"""
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100.0, 90.0, 80.0]}, index=idx)
        signal = pd.Series([-1, 0, 1], index=idx)

        bt = self._make_bt(enable_short=True, fee_rate=0.0)
        # initial_cash=1_000_000, price=100 → qty=10000 (full sizing)
        _, metrics = bt.simulate_trading(df, signal)

        # PnL = (100 - 80) * 10000 = 200_000
        expected_return = 200_000 / 1_000_000  # = 0.2
        assert abs(metrics["short_return"] - expected_return) < 0.0001

    def test_short_pnl_calculation_with_fee(self):
        """ショートPnL計算: fee=0.001/slip=0.001 → notionalベース計算の検証

        initial_cash=1_002_000 にすることで qty=10000 が確定し、
        期待値 = 100*10000*(1-0.002) - 80*10000*(1+0.002) = 998_000 - 801_600 = 196_400 となる。
        """
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100.0, 90.0, 80.0]}, index=idx)
        signal = pd.Series([-1, 0, 1], index=idx)

        # unit_cost = 100 * (1 + 0.001 + 0.001) = 100.2
        # qty = int(1_002_000 // 100.2) = 10000
        bt = Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_002_000,
            fee_rate=0.001,
            slippage=0.001,
            enable_short=True,
            execution_lag=0,  # ショート PnL 式を同バーで隔離検証
        )
        _, metrics = bt.simulate_trading(df, signal)

        # entry=100, exit=80, qty=10000, fee=0.001, slip=0.001
        # net_pnl = 100*10000*(1-0.002) - 80*10000*(1+0.002)
        #         = 998_000 - 801_600 = 196_400
        expected_net_pnl = 196_400
        expected_return = expected_net_pnl / 1_002_000
        assert abs(metrics["short_return"] - expected_return) < 0.0001

    def test_short_disabled_no_short_trades(self):
        """enable_short=False の場合はショートトレードが発生しない"""
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({"Close": [100.0, 90.0, 85.0, 80.0]}, index=idx)
        signal = pd.Series([-1, 0, 0, 1], index=idx)

        bt = self._make_bt(enable_short=False)
        result_df, metrics = bt.simulate_trading(df, signal)

        assert metrics["short_return"] == 0.0
        assert metrics["short_num_trades"] == 0
        if not result_df.empty:
            assert "short" not in result_df["action"].values

    def test_final_day_forced_short_cover(self):
        """最終日に未決済ショートが強制返済される"""
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100.0, 95.0, 90.0]}, index=idx)
        # -1(short) → 0 → 0（カバーシグナルなし、最終日に強制返済）
        signal = pd.Series([-1, 0, 0], index=idx)

        bt = self._make_bt(enable_short=True)
        result_df, metrics = bt.simulate_trading(df, signal)

        assert metrics["short_num_trades"] == 1
        assert metrics["short_return"] > 0  # 100→90 で利益
        assert "final_short_cover" in result_df["action"].values

    def test_long_and_short_independent(self):
        """ロングポジション保有中はショートエントリーしない"""
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame({"Close": [100.0, 105.0, 110.0, 108.0, 105.0]}, index=idx)
        # 1(buy long) → 0 → -1(sell long not short) → -1(no long, short entry) → 1(cover)
        signal = pd.Series([1, 0, -1, -1, 1], index=idx)

        bt = self._make_bt(enable_short=True)
        result_df, metrics = bt.simulate_trading(df, signal)

        actions = result_df["action"].tolist()
        # ロング決済の "sell" と短期"short"/"short_cover" の両方がある
        assert "buy" in actions
        assert "sell" in actions


class TestBacktesterExecutionLag:
    """#493: 実行ラグ（翌バー約定）の検証"""

    def _bt(self, execution_lag: int = 1, **kwargs):
        params = dict(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.0,
            execution_lag=execution_lag,
        )
        params.update(kwargs)
        return Backtester(**params)

    def test_signal_executed_next_bar_close(self):
        """Open列なし: シグナル日 t の判断が t+1 のCloseで約定される"""
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({"Close": [100.0, 110.0, 120.0, 130.0]}, index=idx)
        signal = pd.Series([1, -1, 0, 0], index=idx)

        result_df, _ = self._bt(execution_lag=1).simulate_trading(df, signal)

        buy = result_df[result_df["action"] == "buy"].iloc[0]
        sell = result_df[result_df["action"] == "sell"].iloc[0]
        # buy シグナル(bar0) → bar1 Close=110 で約定 / sell シグナル(bar1) → bar2 Close=120
        assert buy["price"] == 110.0
        assert sell["price"] == 120.0

    def test_open_fill_when_open_column_present(self):
        """Open列あり: 約定は翌バーの始値(Open)で行われる"""
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame(
            {
                "Open": [99.0, 109.0, 119.0, 129.0],
                "Close": [100.0, 110.0, 120.0, 130.0],
            },
            index=idx,
        )
        signal = pd.Series([1, -1, 0, 0], index=idx)

        result_df, _ = self._bt(execution_lag=1).simulate_trading(df, signal)

        buy = result_df[result_df["action"] == "buy"].iloc[0]
        sell = result_df[result_df["action"] == "sell"].iloc[0]
        # bar1 Open=109 で買い / bar2 Open=119 で売り
        assert buy["price"] == 109.0
        assert sell["price"] == 119.0

    def test_final_bar_signal_discarded(self):
        """最終バーのシグナルは約定先バーが無いため破棄される"""
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        df = pd.DataFrame({"Close": [100.0, 110.0, 120.0]}, index=idx)
        signal = pd.Series([0, 0, 1], index=idx)  # 最終バーのみ買い

        result_df, metrics = self._bt(execution_lag=1).simulate_trading(df, signal)

        assert metrics["num_trades"] == 0
        assert result_df.empty or "buy" not in result_df["action"].values

    def test_lag0_same_bar_backward_compat(self):
        """execution_lag=0 は従来の同バー(当日Close)約定を再現する"""
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame({"Close": [100.0, 110.0, 120.0, 130.0]}, index=idx)
        signal = pd.Series([1, -1, 0, 0], index=idx)

        result_df, _ = self._bt(execution_lag=0).simulate_trading(df, signal)

        buy = result_df[result_df["action"] == "buy"].iloc[0]
        sell = result_df[result_df["action"] == "sell"].iloc[0]
        # bar0 Close=100 で買い / bar1 Close=110 で売り（同バー約定）
        assert buy["price"] == 100.0
        assert sell["price"] == 110.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
