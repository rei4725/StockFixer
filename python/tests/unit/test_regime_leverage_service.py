"""ユニットテスト: src.trading.regime_leverage_strategy.service"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch  # noqa: F401  # Task 5/6のテストで使用予定

import numpy as np  # noqa: F401  # Task 5/6のテストで使用予定
import pandas as pd  # noqa: F401  # Task 5/6のテストで使用予定

from src.trading.regime_leverage_strategy.service import (
    compute_equity_now,
    decide_weekly_entry,
    decide_weekly_exit,
)
from src.trading.regime_leverage_strategy.types import RegimeLeverageSnapshot

# Task 5 (decide_daily_check) と Task 6 (run_regime_leverage_*) のテストも
# このファイルに追記される。MagicMock/patch/numpy/pandas はそれらのテストが使う。


def _holding_snapshot(**overrides):
    base = dict(
        id=1,
        executed_at=datetime(2026, 1, 2),
        action="entry",
        reason="regime_entry",
        spy_price_usd=500.0,
        usdjpy_rate=145.0,
        # 自己資金1,000,000円・レバレッジ2.0倍・entry_price_usd=500.5(スリッページ込み)
        # とした場合の株数と整合させる: floor((1,000,000/145.0)*2.0 / 500.5) = 27
        shares=27.0,
        entry_date=datetime(2026, 1, 2),
        entry_price_jpy=72500.0,  # 500.0 * 145.0
        entry_commission_jpy=0.0,
        equity_at_entry_jpy=1_000_000.0,
        stop_price_jpy=65000.0,
        equity_now_jpy=1_000_000.0,
        maintenance_ratio=None,
    )
    base.update(overrides)
    return RegimeLeverageSnapshot(**base)


class TestDecideWeeklyEntry(unittest.TestCase):
    def test_no_entry_when_regime_down(self):
        decision = decide_weekly_entry(
            cash_jpy=1_000_000.0,
            week_close_usd=480.0,
            ma200_usd=500.0,  # close < ma200 → 下降レジーム
            atr14_usd=5.0,
            usdjpy_rate=145.0,
            now=datetime(2026, 1, 9),
        )
        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.reason, "weekly_noop")
        self.assertEqual(decision.shares, 0.0)
        self.assertEqual(decision.equity_now_jpy, 1_000_000.0)

    def test_enters_with_leverage_when_regime_up(self):
        decision = decide_weekly_entry(
            cash_jpy=1_000_000.0,
            week_close_usd=500.0,
            ma200_usd=480.0,  # close > ma200 → 上昇レジーム
            atr14_usd=5.0,
            usdjpy_rate=145.0,
            now=datetime(2026, 1, 9),
        )
        self.assertEqual(decision.action, "entry")
        self.assertEqual(decision.reason, "regime_entry")
        # 建玉USD時価 = 1,000,000 / 145.0 * 2.0 = 13,793.10ドル相当 → shares = floor(13793.10 / entry_price)
        # entry_price_usd = 500.0 * 1.001 = 500.5、entry_price_jpy = 500.5 * 145.0
        self.assertGreater(decision.shares, 0)
        self.assertIsNotNone(decision.entry_price_jpy)
        self.assertIsNotNone(decision.stop_price_jpy)
        self.assertLess(decision.stop_price_jpy, decision.entry_price_jpy)
        # 手計算した期待値を直接検証する(floor divisionのオフバイワンやスリッページ符号反転を検知するため)
        # shares = floor(((1,000,000/145.0)*2.0) / 500.5) = floor(27.5648...) = 27
        # entry_price_jpy = 500.5 * 145.0 = 72,572.5
        # stop_price_jpy = (500.0 - 3.0*5.0) * 145.0 = 485.0 * 145.0 = 70,325.0
        self.assertEqual(decision.shares, 27)
        self.assertAlmostEqual(decision.entry_price_jpy, 72572.5, places=2)
        self.assertAlmostEqual(decision.stop_price_jpy, 70325.0, places=2)


class TestDecideWeeklyExit(unittest.TestCase):
    def test_holds_when_regime_still_up(self):
        snap = _holding_snapshot()
        decision = decide_weekly_exit(
            snap,
            week_close_usd=520.0,
            ma200_usd=480.0,
            usdjpy_rate=146.0,
            now=datetime(2026, 1, 16),
        )
        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.shares, snap.shares)

    def test_exits_on_regime_flip(self):
        snap = _holding_snapshot()
        decision = decide_weekly_exit(
            snap,
            week_close_usd=470.0,
            ma200_usd=480.0,
            usdjpy_rate=146.0,
            now=datetime(2026, 1, 16),
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "regime_flip")
        self.assertEqual(decision.shares, 0.0)
        # exit時のequity_now_jpyをcompute_equity_nowと同じ計算式で手計算し検証する
        # current_price_jpy = 470.0 * 146.0 = 68,620.0
        # exit_price_jpy = 68,620.0 * (1 - 0.001) = 68,551.38 (売却スリッページ)
        # unrealized_pnl = (68,551.38 - 72,500.0) * 27 = -106,612.74
        # days_held = (2026-01-16 - 2026-01-02).days = 14
        # interest_accrued = 72,500.0 * 27 * 0.030 / 365 * 14 = 2,252.4657...
        # equity = 1,000,000.0 - 106,612.74 - 2,252.4657... - 0.0 = 891,134.79
        current_price_jpy = 470.0 * 146.0
        exit_price_jpy = current_price_jpy * (1 - 0.001)
        expected_unrealized = (exit_price_jpy - snap.entry_price_jpy) * snap.shares
        days_held = (datetime(2026, 1, 16).date() - snap.entry_date.date()).days
        expected_interest = snap.entry_price_jpy * snap.shares * 0.030 / 365 * days_held
        expected_equity = (
            snap.equity_at_entry_jpy
            + expected_unrealized
            - expected_interest
            - snap.entry_commission_jpy
        )
        self.assertAlmostEqual(decision.equity_now_jpy, expected_equity, places=2)
        self.assertAlmostEqual(decision.equity_now_jpy, 891134.79, places=2)


class TestComputeEquityNow(unittest.TestCase):
    def test_flat_price_returns_equity_at_entry_minus_interest(self):
        snap = _holding_snapshot(entry_commission_jpy=0.0)
        equity = compute_equity_now(
            snap, current_price_jpy=snap.entry_price_jpy, now=datetime(2026, 1, 3)
        )
        # 1日分の金利のみ差し引かれる: entry_price_jpy * shares * 0.03 / 365 * 1日
        expected_interest = snap.entry_price_jpy * snap.shares * 0.030 / 365 * 1
        self.assertAlmostEqual(equity, snap.equity_at_entry_jpy - expected_interest, places=2)

    def test_price_rise_increases_equity(self):
        snap = _holding_snapshot(entry_commission_jpy=0.0)
        higher_price = snap.entry_price_jpy * 1.05
        equity = compute_equity_now(snap, current_price_jpy=higher_price, now=snap.entry_date)
        expected_unrealized = (higher_price - snap.entry_price_jpy) * snap.shares
        self.assertAlmostEqual(equity, snap.equity_at_entry_jpy + expected_unrealized, places=2)


class TestDecideDailyCheck(unittest.TestCase):
    def test_noop_when_above_stop_and_maintenance(self):
        from src.trading.regime_leverage_strategy.service import decide_daily_check

        snap = _holding_snapshot()
        decision = decide_daily_check(
            snap, day_low_usd=510.0, usdjpy_rate=146.0, now=datetime(2026, 1, 5)
        )
        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.reason, "daily_noop")
        self.assertEqual(decision.shares, snap.shares)
        self.assertIsNotNone(decision.maintenance_ratio)
        # 手計算した期待値を直接検証する(維持率が閾値0.20を大きく上回ることを確認するため)
        # day_low_jpy = 510.0 * 146.0 = 74,460.0
        # unrealized_pnl = (74,460.0 - 72,500.0) * 27 = 52,920.0
        # days_held = (2026-01-05 - 2026-01-02).days = 3
        # interest_accrued = 72,500.0 * 27 * 0.030 / 365 * 3 = 482.6712...
        # equity_at_low = 1,000,000.0 + 52,920.0 - 482.6712... - 0.0 = 1,052,437.3288
        # value_at_low = 74,460.0 * 27 = 2,010,420.0
        # maintenance_ratio = 1,052,437.3288 / 2,010,420.0 = 0.523491...
        self.assertAlmostEqual(decision.equity_now_jpy, 1052437.3288, places=2)
        self.assertAlmostEqual(decision.maintenance_ratio, 0.5234912748, places=6)

    def test_initial_stop_triggers_exit(self):
        from src.trading.regime_leverage_strategy.service import decide_daily_check

        snap = _holding_snapshot(stop_price_jpy=70000.0)
        # day_low_usdを円換算するとstop_price_jpy(70000)を下回る値にする
        decision = decide_daily_check(
            snap, day_low_usd=470.0, usdjpy_rate=145.0, now=datetime(2026, 1, 5)
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "initial_stop")
        self.assertEqual(decision.shares, 0.0)
        # 維持率は0.20を上回っており(margin_callではない)、initial_stopの優先順位が
        # 正しく2番目に判定されたことを確認する
        # day_low_jpy = 470.0 * 145.0 = 68,150.0
        # maintenance_ratio(day_low基準) = 882,067.3288 / 1,840,050.0 = 0.479371...
        self.assertAlmostEqual(decision.maintenance_ratio, 0.4793713914, places=6)
        self.assertGreater(decision.maintenance_ratio, 0.20)
        # exit_price_jpy = stop_price_jpy(70,000.0) * (1 - 0.001) = 69,930.0 (損切りのスリッページ)
        # unrealized_pnl = (69,930.0 - 72,500.0) * 27 = -69,390.0
        # interest_accrued = 482.6712...(上と同じ, days_held=3)
        # equity = 1,000,000.0 - 69,390.0 - 482.6712... - 0.0 = 930,127.3288
        self.assertAlmostEqual(decision.equity_now_jpy, 930127.3288, places=2)

    def test_margin_call_triggers_exit_before_stop_check(self):
        from src.trading.regime_leverage_strategy.service import decide_daily_check

        # レバレッジ2倍で建てた直後に急落し、維持率が0.20を割るケース
        snap = _holding_snapshot(
            shares=4000.0,
            entry_price_jpy=72500.0,
            equity_at_entry_jpy=1_000_000.0,
            entry_commission_jpy=0.0,
            stop_price_jpy=1000.0,  # stopには触れない値(day_low_jpy=29,000.0 > 1,000.0)
        )
        decision = decide_daily_check(
            snap, day_low_usd=200.0, usdjpy_rate=145.0, now=datetime(2026, 1, 5)
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "margin_call")
        # 維持率がマイナスまで急落しており、0.20の閾値を明確に下回ることを確認する
        # (initial_stopではなくmargin_callが優先されたことの裏付け)
        # day_low_jpy = 200.0 * 145.0 = 29,000.0
        # value_at_low = 29,000.0 * 4,000 = 116,000,000.0
        # equity_at_low = 1,000,000.0 + (29,000.0-72,500.0)*4,000 - interest = -173,071,506.8493...
        # maintenance_ratio = -173,071,506.8493... / 116,000,000.0 = -1.491996...
        self.assertAlmostEqual(decision.maintenance_ratio, -1.491995749, places=6)
        self.assertLess(decision.maintenance_ratio, 0.20)

    def test_margin_call_wins_over_stop_when_both_conditions_true(self):
        """両方の条件が同時に真になるフィクスチャで、margin_callが優先されることを検証する。

        test_margin_call_triggers_exit_before_stop_check は stop_price_jpy=1000.0 と
        いう、初期損切り条件(day_low_jpy <= stop_price_jpy)がそもそも真にならない値を
        使っているため、判定順序が逆転(初期損切りを先にチェック)しても検知できない。
        このテストでは stop_price_jpy=50000.0 とし、day_low_jpy(29,000.0) <= 50,000.0 も
        真にする。両条件が真の状態でも margin_call が返ること、かつ exit 価格が
        day_low_jpy(margin_call側)を使っており stop_price_jpy(initial_stop側)を
        使っていないことを equity_now_jpy の厳密値で確認する。
        """
        from src.trading.regime_leverage_strategy.service import decide_daily_check

        snap = _holding_snapshot(
            shares=4000.0,
            entry_price_jpy=72500.0,
            equity_at_entry_jpy=1_000_000.0,
            entry_commission_jpy=0.0,
            stop_price_jpy=50000.0,  # day_low_jpy(29,000.0) <= 50,000.0 も真になる値
        )
        decision = decide_daily_check(
            snap, day_low_usd=200.0, usdjpy_rate=145.0, now=datetime(2026, 1, 5)
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "margin_call")
        self.assertLess(decision.maintenance_ratio, 0.20)
        # day_low_jpy = 200.0 * 145.0 = 29,000.0 (<= stop_price_jpy=50,000.0 も真だが、
        # margin_callが先にreturnするためinitial_stop分岐へは到達しないはず)
        # exit_price_jpy(margin_call) = 29,000.0 * (1 - 0.001) = 28,971.0
        # equity = 1,000,000.0 + (28,971.0-72,500.0)*4,000 - interest - 0.0 = -173,187,506.8493
        # もしinitial_stopが誤って先に判定されると
        # exit_price_jpy = 50,000.0*(1-0.001) = 49,950.0 となり
        # equity = -89,271,506.8493 という全く異なる値になるため、
        # この厳密値アサーションは優先順位バグを確実に検知できる
        self.assertAlmostEqual(decision.equity_now_jpy, -173187506.8493, places=1)


class TestRunRegimeLeverageWeeklyCheck(unittest.TestCase):
    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_first_run_uses_initial_capital_and_enters_on_uptrend(self, mock_latest, mock_insert):
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        mock_latest.return_value = None
        mock_port = MagicMock()
        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(400.0, 500.0, 260), index=idx)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 2,
                "Low": prices - 2,
                "Close": prices,
                "Volume": 1_000_000,
            },
            index=idx,
        )
        mock_port.get_stock_data.return_value = df
        fx_df = pd.DataFrame({"Close": [145.0]}, index=[idx[-1]])
        mock_port.get_forex_data.return_value = fx_df

        decision = run_regime_leverage_weekly_check(mock_port)

        self.assertEqual(decision.action, "entry")
        # 初回実行(snapshot=None)では自己資金としてREGIME_LEVERAGE_INITIAL_CAPITAL_JPY
        # (1,000,000)が使われるべき。誤った資金源(例: 0円やハードコードされた別の値)が
        # 使われていても検知できるよう、equity_at_entry_jpyを厳密に検証する。
        self.assertEqual(decision.equity_at_entry_jpy, 1_000_000.0)
        mock_insert.assert_called_once()

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_reuses_previous_equity_now_after_closed_position(self, mock_latest, mock_insert):
        """2回目以降(直前のポジションが決済済み: shares=0)は、直前snapshotの
        equity_now_jpyを新規エントリーの自己資金として使うべき
        (REGIME_LEVERAGE_INITIAL_CAPITAL_JPYに戻ってはいけない)。
        """
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        mock_latest.return_value = _holding_snapshot(shares=0.0, equity_now_jpy=1_200_000.0)
        mock_port = MagicMock()
        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(400.0, 500.0, 260), index=idx)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 2,
                "Low": prices - 2,
                "Close": prices,
                "Volume": 1_000_000,
            },
            index=idx,
        )
        mock_port.get_stock_data.return_value = df
        fx_df = pd.DataFrame({"Close": [145.0]}, index=[idx[-1]])
        mock_port.get_forex_data.return_value = fx_df

        decision = run_regime_leverage_weekly_check(mock_port)

        self.assertEqual(decision.action, "entry")
        self.assertEqual(decision.equity_at_entry_jpy, 1_200_000.0)
        mock_insert.assert_called_once()

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_holding_and_regime_still_up_calls_decide_weekly_exit_as_noop(
        self, mock_latest, mock_insert
    ):
        """保有中(shares>0)の分岐ではdecide_weekly_exitが呼ばれるべき。
        レジームが上昇継続の週足を与え、noopで既存ポジションが維持されることを検証する。
        """
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        snap = _holding_snapshot()
        mock_latest.return_value = snap
        mock_port = MagicMock()
        # 200日線を大きく上回る右肩上がりの価格系列(レジーム上昇継続)
        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(400.0, 600.0, 260), index=idx)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 2,
                "Low": prices - 2,
                "Close": prices,
                "Volume": 1_000_000,
            },
            index=idx,
        )
        mock_port.get_stock_data.return_value = df
        fx_df = pd.DataFrame({"Close": [146.0]}, index=[idx[-1]])
        mock_port.get_forex_data.return_value = fx_df

        decision = run_regime_leverage_weekly_check(mock_port)

        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.reason, "weekly_noop")
        self.assertEqual(decision.shares, snap.shares)
        mock_insert.assert_called_once()

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_holding_and_regime_down_calls_decide_weekly_exit_as_exit(
        self, mock_latest, mock_insert
    ):
        """保有中(shares>0)でレジームが下降転換した週足を与えると、decide_weekly_exit
        経由でexitが返ることを検証する(holding分岐がdecide_weekly_exitを正しく
        呼んでいることの裏付け)。
        """
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        snap = _holding_snapshot()
        mock_latest.return_value = snap
        mock_port = MagicMock()
        # 200日線を大きく下回る右肩下がりの価格系列(レジーム下降転換)
        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(600.0, 400.0, 260), index=idx)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 2,
                "Low": prices - 2,
                "Close": prices,
                "Volume": 1_000_000,
            },
            index=idx,
        )
        mock_port.get_stock_data.return_value = df
        fx_df = pd.DataFrame({"Close": [146.0]}, index=[idx[-1]])
        mock_port.get_forex_data.return_value = fx_df

        decision = run_regime_leverage_weekly_check(mock_port)

        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "regime_flip")
        self.assertEqual(decision.shares, 0.0)
        mock_insert.assert_called_once()


class TestRunRegimeLeverageDailyMarginCheck(unittest.TestCase):
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_returns_none_when_not_holding(self, mock_latest):
        from src.trading.regime_leverage_strategy.service import (
            run_regime_leverage_daily_margin_check,
        )

        mock_latest.return_value = None
        mock_port = MagicMock()
        result = run_regime_leverage_daily_margin_check(mock_port)
        self.assertIsNone(result)
        mock_port.get_stock_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
