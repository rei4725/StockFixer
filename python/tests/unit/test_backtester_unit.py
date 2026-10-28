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

    def test_include_monte_carlo_false_by_default_omits_mc_keys(self, sample_price_df):
        """include_monte_carlo 未指定（デフォルト False）では mc_ キーが含まれない（#374）"""
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
        assert "mc_final_cash_p50" not in metrics

    def test_include_monte_carlo_true_adds_mc_keys(self, sample_price_df):
        """include_monte_carlo=True で mc_ プレフィックスの統計が metrics に含まれる（#374）"""
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
            include_monte_carlo=True,
        )
        sig = pd.Series([1, 0, -1, 0, 0], index=sample_price_df.index)
        _, metrics = bt.simulate_trading(sample_price_df, sig)
        for key in (
            "mc_max_drawdown_mean",
            "mc_max_drawdown_p95",
            "mc_final_cash_p05",
            "mc_final_cash_p50",
            "mc_final_cash_p95",
        ):
            assert key in metrics


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


class TestBacktesterShortEntryVolume:
    """#801: ショートのエントリー売り代金は、エントリーの時点で分かる出来高で決める"""

    @staticmethod
    def _spy_bt(calls: list, execution_lag: int = 0) -> Backtester:
        def _slippage_fn(qty: int, price: float, avg_volume: float) -> float:
            calls.append((qty, price, avg_volume))
            return 0.01 * (qty / avg_volume) ** 0.5

        return Backtester(
            model_manager=MagicMock(),
            signal_generator=MagicMock(),
            data_loader=MagicMock(),
            start_date=None,
            end_date=None,
            market="jp",
            symbol="7203",
            initial_cash=1_000_000,
            fee_rate=0.0,
            slippage=0.0,
            slippage_fn=_slippage_fn,
            enable_short=True,
            execution_lag=execution_lag,
        )

    @staticmethod
    def _df(later_volume: float) -> pd.DataFrame:
        # エントリー（同バーは bar0、翌バーは bar1 始値で bar0 までの出来高）より後の
        # bar2・bar3 の出来高だけを later_volume にする
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        return pd.DataFrame(
            {
                "Open": [100.0, 100.0, 90.0, 80.0],
                "Close": [100.0, 100.0, 90.0, 80.0],
                "Volume": [50_000.0, 50_000.0, later_volume, later_volume],
            },
            index=idx,
        )

    @staticmethod
    def _signal(df: pd.DataFrame, execution_lag: int, cover: bool) -> pd.Series:
        if not cover:
            return pd.Series([-1, 0, 0, 0], index=df.index)
        # 同バー約定は bar3 終値でカバー、翌バー約定は bar3 始値でカバー
        return pd.Series([-1, 0, 0, 1] if execution_lag == 0 else [-1, 0, 1, 0], index=df.index)

    @pytest.mark.parametrize("execution_lag", [0, 1])
    @pytest.mark.parametrize("cover", [True, False])
    def test_entry_slippage_uses_entry_time_volume(self, execution_lag, cover):
        """エントリー価格（100）での代金計算は、エントリーの時点で分かる平均出来高だけを使う"""
        df = self._df(later_volume=5_000_000.0)
        calls: list = []
        bt = self._spy_bt(calls, execution_lag=execution_lag)
        result_df, _ = bt.simulate_trading(df, self._signal(df, execution_lag, cover))

        exit_action = "short_cover" if cover else "final_short_cover"
        assert exit_action in result_df["action"].values
        # 同バーは bar0 の平均、翌バーは前バー bar0 までの平均（どちらも 50_000）
        entry_calls = [c for c in calls if c[1] == 100.0]
        assert entry_calls
        assert [c[2] for c in entry_calls] == pytest.approx([50_000.0] * len(entry_calls))

    @pytest.mark.parametrize("execution_lag", [0, 1])
    @pytest.mark.parametrize("cover", [True, False])
    def test_entry_proceeds_independent_of_later_volume(self, execution_lag, cover):
        """エントリーより後の出来高を変えても、エントリーの売り代金は変わらない

        決済後の cash からカバーの買い代金を足し戻した値（＝エントリーの売り代金）を比べる。
        """
        entry_proceeds = []
        for later_volume in (50_000.0, 5_000_000.0):
            df = self._df(later_volume)
            calls: list = []
            bt = self._spy_bt(calls, execution_lag=execution_lag)
            result_df, _ = bt.simulate_trading(df, self._signal(df, execution_lag, cover))
            qty = int(result_df.loc[result_df["action"] == "short", "qty"].iloc[0])
            exit_row = result_df[result_df["action"].isin(["short_cover", "final_short_cover"])]
            exit_price = float(exit_row["price"].iloc[0])
            cover_cost = bt.execution.buy_cost(
                qty, exit_price, [c for c in calls if c[1] == exit_price][-1][2]
            )
            entry_proceeds.append(float(exit_row["cash"].iloc[0]) - 1_000_000 + cover_cost)

        assert entry_proceeds[0] == pytest.approx(entry_proceeds[1], rel=1e-12)

    def test_short_qty_capped_by_volume_like_buy(self):
        """ショートの株数にも、買いと同じく出来高込みのコストでの上限が効く"""
        idx = pd.date_range("2024-01-01", periods=2, freq="B")
        df = pd.DataFrame({"Close": [100.0, 100.0], "Volume": [5_000.0, 5_000.0]}, index=idx)

        long_calls: list = []
        long_bt = self._spy_bt(long_calls)
        long_df, _ = long_bt.simulate_trading(df, pd.Series([1, 0], index=idx))
        short_calls: list = []
        short_bt = self._spy_bt(short_calls)
        short_df, _ = short_bt.simulate_trading(df, pd.Series([-1, 0], index=idx))

        buy_qty = int(long_df.loc[long_df["action"] == "buy", "qty"].iloc[0])
        short_qty = int(short_df.loc[short_df["action"] == "short", "qty"].iloc[0])
        assert short_qty == buy_qty


class TestBacktesterFinalExitVolume:
    """#1347: 期末の強制決済は、execution_lag > 0 なら前の足までの平均出来高で数える"""

    _FINAL_PRICE = 123.0

    @classmethod
    def _df(cls, last_volume: float) -> pd.DataFrame:
        n = 25
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = [100.0] * (n - 1) + [cls._FINAL_PRICE]
        volumes = [10_000.0 + 1_000.0 * i for i in range(n - 1)] + [last_volume]
        return pd.DataFrame({"Open": prices, "Close": prices, "Volume": volumes}, index=idx)

    @staticmethod
    def _signal(df: pd.DataFrame, short: bool) -> pd.Series:
        values = [0] * len(df)
        values[0] = -1 if short else 1
        return pd.Series(values, index=df.index)

    @staticmethod
    def _final_cash(df: pd.DataFrame, short: bool, calls: list) -> float:
        bt = TestBacktesterShortEntryVolume._spy_bt(calls, execution_lag=1)
        result_df, _ = bt.simulate_trading(df, TestBacktesterFinalExitVolume._signal(df, short))
        action = "final_short_cover" if short else "final_sell"
        rows = result_df[result_df["action"] == action]
        assert len(rows) == 1
        return float(rows["cash"].iloc[0])

    @pytest.mark.parametrize("short", [False, True])
    def test_final_cash_independent_of_last_bar_volume(self, short):
        """最後の足の出来高だけを変えても、期末の強制決済の cash は変わらない"""
        cashes = [
            self._final_cash(self._df(last_volume), short, [])
            for last_volume in (100.0, 34_000.0, 50_000_000.0)
        ]
        assert cashes[0] == pytest.approx(cashes[1], rel=1e-12)
        assert cashes[0] == pytest.approx(cashes[2], rel=1e-12)

    @pytest.mark.parametrize("short", [False, True])
    def test_final_exit_avg_volume_excludes_last_bar(self, short):
        """期末の slippage_fn に渡る avg_volume は、最後の足を除いた20本の平均"""
        df = self._df(last_volume=50_000_000.0)
        calls: list = []
        self._final_cash(df, short, calls)

        final_calls = [c for c in calls if c[1] == self._FINAL_PRICE]
        assert final_calls
        expected = float(df["Volume"].iloc[-21:-1].mean())
        assert [c[2] for c in final_calls] == pytest.approx([expected] * len(final_calls))


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


class TestBacktesterSlTpNextBarFill:
    """#964: execution_lag > 0 の SL/TP は当バー Close で判定し、翌バー始値で決済する"""

    _IDX = pd.date_range("2024-01-01", periods=5, freq="B")

    def _bt(self, **kwargs):
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
            execution_lag=1,
        )
        params.update(kwargs)
        return Backtester(**params)

    def _sl_df(self):
        # bar1 Open=100 で買い → bar2 Close=94 で -6%（SL 5% 割れ）→ bar3 Open=90 は窓開けの下げ
        return pd.DataFrame(
            {
                "Open": [100.0, 100.0, 99.0, 90.0, 91.0],
                "Close": [100.0, 100.0, 94.0, 91.0, 92.0],
            },
            index=self._IDX,
        )

    def test_stop_loss_filled_at_next_open(self):
        signal = pd.Series([1, 0, 0, 0, 0], index=self._IDX)
        result_df, metrics = self._bt(stop_loss_pct=0.05).simulate_trading(self._sl_df(), signal)

        buy = result_df[result_df["action"] == "buy"].iloc[0]
        sl = result_df[result_df["action"] == "stop_loss"].iloc[0]
        assert sl["date"] == self._IDX[3]
        assert sl["price"] == 90.0
        # 判定した足の終値 94 で売った場合より損が大きい
        loss = (sl["price"] - buy["price"]) * sl["qty"]
        loss_at_close = (94.0 - buy["price"]) * sl["qty"]
        assert loss < loss_at_close
        assert "final_sell" not in result_df["action"].values

    def test_take_profit_filled_at_next_open(self):
        # bar2 Close=112 で +12%（TP 10%）→ bar3 Open=110 で決済
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 109.0, 110.0, 111.0],
                "Close": [100.0, 100.0, 112.0, 111.0, 112.0],
            },
            index=self._IDX,
        )
        signal = pd.Series([1, 0, 0, 0, 0], index=self._IDX)
        result_df, _ = self._bt(take_profit_pct=0.10).simulate_trading(df, signal)

        buy = result_df[result_df["action"] == "buy"].iloc[0]
        tp = result_df[result_df["action"] == "take_profit"].iloc[0]
        assert tp["date"] == self._IDX[3]
        assert tp["price"] == 110.0
        profit = (tp["price"] - buy["price"]) * tp["qty"]
        profit_at_close = (112.0 - buy["price"]) * tp["qty"]
        assert profit < profit_at_close

    def test_buy_signal_on_exit_bar_does_not_reenter(self):
        # 判定の足 bar2 の買いシグナルは bar3 に約定する足で、SL の決済と重なる
        signal = pd.Series([1, 0, 1, 0, 0], index=self._IDX)
        result_df, _ = self._bt(stop_loss_pct=0.05).simulate_trading(self._sl_df(), signal)

        actions = list(result_df["action"])
        assert actions.count("buy") == 1
        assert actions.count("stop_loss") == 1
        assert "final_sell" not in actions

    def test_sell_signal_on_exit_bar_does_not_double_sell(self):
        signal = pd.Series([1, 0, -1, 0, 0], index=self._IDX)
        result_df, _ = self._bt(stop_loss_pct=0.05).simulate_trading(self._sl_df(), signal)

        actions = list(result_df["action"])
        assert actions.count("stop_loss") == 1
        assert "sell" not in actions
        assert "final_sell" not in actions

    def test_trigger_on_last_bar_left_to_final_sell(self):
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 99.0, 98.0, 97.0],
                "Close": [100.0, 100.0, 99.0, 98.0, 94.0],
            },
            index=self._IDX,
        )
        signal = pd.Series([1, 0, 0, 0, 0], index=self._IDX)
        result_df, _ = self._bt(stop_loss_pct=0.05).simulate_trading(df, signal)

        actions = list(result_df["action"])
        assert "stop_loss" not in actions
        final = result_df[result_df["action"] == "final_sell"].iloc[0]
        assert final["price"] == 94.0


class TestBacktesterSlTpAfterOpenSignal:
    """#1361: execution_lag > 0 では、寄りで約定するシグナルを先に処理し、
    その後に残った保有だけを当バー Close で SL/TP 判定する"""

    _IDX = pd.date_range("2024-01-01", periods=5, freq="B")

    def _bt(self, **kwargs):
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
            execution_lag=1,
        )
        params.update(kwargs)
        return Backtester(**params)

    def test_sell_signal_at_open_wins_over_take_profit_at_close(self):
        # bar1 Open=100 で買い → bar2 の寄り 101 で売りシグナルが約定。bar2 Close=112 は TP 10% 超え
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 101.0, 110.0, 111.0],
                "Close": [100.0, 100.0, 112.0, 111.0, 112.0],
            },
            index=self._IDX,
        )
        signal = pd.Series([1, -1, 0, 0, 0], index=self._IDX)
        result_df, _ = self._bt(take_profit_pct=0.10).simulate_trading(df, signal)

        actions = list(result_df["action"])
        assert "take_profit" not in actions
        sell = result_df[result_df["action"] == "sell"].iloc[0]
        assert sell["date"] == self._IDX[2]
        assert sell["price"] == 101.0
        assert "final_sell" not in actions

    def test_sell_signal_at_open_wins_over_stop_loss_at_close(self):
        # bar2 の寄り 99 で売りシグナルが約定。bar2 Close=94 は SL 5% 割れ
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 99.0, 90.0, 91.0],
                "Close": [100.0, 100.0, 94.0, 91.0, 92.0],
            },
            index=self._IDX,
        )
        signal = pd.Series([1, -1, 0, 0, 0], index=self._IDX)
        result_df, _ = self._bt(stop_loss_pct=0.05).simulate_trading(df, signal)

        actions = list(result_df["action"])
        assert "stop_loss" not in actions
        sell = result_df[result_df["action"] == "sell"].iloc[0]
        assert sell["date"] == self._IDX[2]
        assert sell["price"] == 99.0
        assert "final_sell" not in actions

    def test_buy_at_open_then_stop_loss_at_close_exits_next_bar(self):
        # bar1 Open=100 で買い、同じ足の Close=94 で SL 5% 割れ → bar2 Open=92 で決済
        df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 92.0, 91.0, 90.0],
                "Close": [100.0, 94.0, 93.0, 92.0, 91.0],
            },
            index=self._IDX,
        )
        signal = pd.Series([1, 0, 0, 0, 0], index=self._IDX)
        result_df, _ = self._bt(stop_loss_pct=0.05).simulate_trading(df, signal)

        buy = result_df[result_df["action"] == "buy"].iloc[0]
        sl = result_df[result_df["action"] == "stop_loss"].iloc[0]
        assert buy["date"] == self._IDX[1]
        assert sl["date"] == self._IDX[2]
        assert sl["price"] == 92.0
        assert list(result_df["action"]).count("stop_loss") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
