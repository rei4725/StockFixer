"""ユニットテスト: src.trading.regime_leverage_strategy.service"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch  # noqa: F401  # Task 5/6のテストで使用予定
from zoneinfo import ZoneInfo

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

    @patch(
        "src.trading.regime_leverage_strategy.service.get_last_daily_check_at",
        return_value=None,
    )
    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_holding_and_above_stop_and_maintenance_calls_decide_daily_check_as_noop(
        self, mock_latest, mock_insert, _mock_last_daily
    ):
        """保有中(shares>0)で当日安値がstop_price_jpyにも維持率にも抵触しない場合、
        decide_daily_checkがnoopを返し、その結果がinsert_snapshotへ渡されることを検証する
        (これまでholding分岐は未保有の早期returnしかテストされていなかった)。
        """
        from src.trading.regime_leverage_strategy.service import (
            run_regime_leverage_daily_margin_check,
        )

        snap = _holding_snapshot()  # stop_price_jpy=65000.0, shares=27.0
        mock_latest.return_value = snap
        mock_port = MagicMock()
        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(400.0, 500.0, 260), index=idx)
        # 最終行(当日安値)を510.0ドルにする: day_low_jpy = 510.0*146.0 = 74,460.0
        # (stop_price_jpy=65,000.0を大きく上回り、維持率も0.20を大きく上回るため
        # margin_call/initial_stopいずれにも抵触しない = noopとなるはず)
        lows = prices - 2
        lows.iloc[-1] = 510.0
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 2,
                "Low": lows,
                "Close": prices,
                "Volume": 1_000_000,
            },
            index=idx,
        )
        mock_port.get_stock_data.return_value = df
        fx_df = pd.DataFrame({"Close": [146.0]}, index=[idx[-1]])
        mock_port.get_forex_data.return_value = fx_df

        decision = run_regime_leverage_daily_margin_check(mock_port)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.reason, "daily_noop")
        self.assertEqual(decision.shares, snap.shares)
        self.assertIsNotNone(decision.maintenance_ratio)
        self.assertGreater(decision.maintenance_ratio, 0.20)
        mock_insert.assert_called_once_with(decision)

    @patch(
        "src.trading.regime_leverage_strategy.service.get_last_daily_check_at",
        return_value=None,
    )
    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_raises_and_skips_insert_when_holding_and_day_low_is_nan(
        self, mock_latest, mock_insert, _mock_last_daily
    ):
        """保有中に当日安値(Low)がNaN(データ劣化時)の場合、判定を行わず例外を送出し、DBにも
        書き込まないこと(NaN比較は常にFalseになり誤ってnoop/exitと判定してしまうため)。
        未保有のNone(何もしない)と区別し、スケジューラーのリトライ・Discord通知に乗せる(#860)。
        """
        from src.trading.regime_leverage_strategy.service import (
            run_regime_leverage_daily_margin_check,
        )

        mock_latest.return_value = _holding_snapshot()
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
        df.loc[df.index[-1], "Low"] = np.nan
        mock_port.get_stock_data.return_value = df
        fx_df = pd.DataFrame({"Close": [145.0]}, index=[idx[-1]])
        mock_port.get_forex_data.return_value = fx_df

        with self.assertRaises(RuntimeError):
            run_regime_leverage_daily_margin_check(mock_port)

        mock_insert.assert_not_called()


class TestRegimeLeverageNaNGuard(unittest.TestCase):
    """Finding 2: MA200/ATR14がNaNの場合、判定を行わずDBにも書き込まないこと。
    未保有ならNoneを返し、保有中なら例外を送出する(#870)。"""

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_weekly_check_returns_none_when_ma200_is_nan(self, mock_latest, mock_insert):
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        mock_latest.return_value = None
        mock_port = MagicMock()
        # 200日線が計算できるだけの行数(200)に満たないデータ(150行)を与え、
        # 最終行のMA200がNaNになるようにする。
        idx = pd.bdate_range("2025-01-01", periods=150)
        prices = pd.Series(np.linspace(400.0, 500.0, 150), index=idx)
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

        self.assertIsNone(decision)
        mock_insert.assert_not_called()

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_weekly_check_holding_raises_when_ma200_is_nan_and_does_not_liquidate(
        self, mock_latest, mock_insert
    ):
        """保有中にMA200がNaNになった場合、誤って regime_flip 決済せず、判定が黙って
        飛ばないよう例外を送出する(スケジューラーのリトライ・Discord通知に乗せる。#870)。
        """
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        mock_latest.return_value = _holding_snapshot()
        mock_port = MagicMock()
        idx = pd.bdate_range("2025-01-01", periods=150)
        prices = pd.Series(np.linspace(400.0, 500.0, 150), index=idx)
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

        with self.assertRaises(RuntimeError):
            run_regime_leverage_weekly_check(mock_port)

        mock_insert.assert_not_called()

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_weekly_check_holding_raises_when_atr14_is_nan(self, mock_latest, mock_insert):
        """保有中にATR14がNaN(最終行の高値・安値が欠損)の場合も例外を送出する(#870)。"""
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        mock_latest.return_value = _holding_snapshot()
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
        df.loc[df.index[-1], ["High", "Low"]] = np.nan
        mock_port.get_stock_data.return_value = df
        fx_df = pd.DataFrame({"Close": [145.0]}, index=[idx[-1]])
        mock_port.get_forex_data.return_value = fx_df

        with self.assertRaises(RuntimeError):
            run_regime_leverage_weekly_check(mock_port)

        mock_insert.assert_not_called()


class TestRegimeLeverageUsdJpyGuard(unittest.TestCase):
    """ドル円の最終行の終値がNaNまたは0以下の場合、判定を行わず例外を送出し、
    NaNの評価額をDBに書き込まないこと(#871)。"""

    @staticmethod
    def _port_with_fx(fx_close: float, prices: pd.Series) -> MagicMock:
        idx = prices.index
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
        mock_port = MagicMock()
        mock_port.get_stock_data.return_value = df
        mock_port.get_forex_data.return_value = pd.DataFrame(
            {"Close": [145.0, fx_close]}, index=[idx[-2], idx[-1]]
        )
        return mock_port

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_daily_check_holding_raises_when_usdjpy_is_nan(self, mock_latest, mock_insert):
        from src.trading.regime_leverage_strategy.service import (
            run_regime_leverage_daily_margin_check,
        )

        mock_latest.return_value = _holding_snapshot()
        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(400.0, 500.0, 260), index=idx)
        mock_port = self._port_with_fx(np.nan, prices)

        with self.assertRaises(RuntimeError):
            run_regime_leverage_daily_margin_check(mock_port)

        mock_insert.assert_not_called()

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_daily_check_holding_raises_when_usdjpy_is_zero(self, mock_latest, mock_insert):
        from src.trading.regime_leverage_strategy.service import (
            run_regime_leverage_daily_margin_check,
        )

        mock_latest.return_value = _holding_snapshot()
        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(400.0, 500.0, 260), index=idx)
        mock_port = self._port_with_fx(0.0, prices)

        with self.assertRaises(RuntimeError):
            run_regime_leverage_daily_margin_check(mock_port)

        mock_insert.assert_not_called()

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_weekly_check_holding_raises_when_usdjpy_is_nan(self, mock_latest, mock_insert):
        """保有中にレジームが下降転換した週でも、ドル円がNaNならNaNの評価額で
        regime_flipのexit行を書かずに例外を送出する。"""
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        mock_latest.return_value = _holding_snapshot()
        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(600.0, 400.0, 260), index=idx)
        mock_port = self._port_with_fx(np.nan, prices)

        with self.assertRaises(RuntimeError):
            run_regime_leverage_weekly_check(mock_port)

        mock_insert.assert_not_called()


_JST = ZoneInfo("Asia/Tokyo")


def _fixed_datetime(now_jst: datetime):
    """service.datetime を差し替え、now() だけを固定の JST の時刻にする。"""

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return now_jst.replace(tzinfo=None)
            return now_jst.astimezone(tz)

    return _FixedDatetime


def _daily_port(last_bar: str, lows: dict) -> MagicMock:
    """last_bar の日までの日足(安値は既定で 498 ドル、lows で日ごとに上書き)を返すポート。"""
    idx = pd.bdate_range(end=last_bar, periods=260)
    prices = pd.Series(500.0, index=idx)
    low = prices - 2
    for day, value in lows.items():
        low.loc[pd.Timestamp(day)] = value
    df = pd.DataFrame(
        {"Open": prices, "High": prices + 2, "Low": low, "Close": prices, "Volume": 1_000_000},
        index=idx,
    )
    mock_port = MagicMock()
    mock_port.get_stock_data.return_value = df
    mock_port.get_forex_data.return_value = pd.DataFrame({"Close": [146.0]}, index=[idx[-1]])
    return mock_port


class TestDailyCheckUsesClosedBarsSinceLastRun(unittest.TestCase):
    """日次判定は、前の回の後に引けた日足を全部見て、引けていない当日の足は使わない(#891)。
    stop_price_jpy=65,000円・ドル円146円なので、安値445ドル以下で損切りの値段を割る。"""

    def _run(
        self,
        now_jst: datetime,
        executed_at: datetime,
        port: MagicMock,
        reason: str = "daily_noop",
        last_daily_check_at=None,
    ):
        from src.trading.regime_leverage_strategy.service import (
            run_regime_leverage_daily_margin_check,
        )

        snap = _holding_snapshot(
            executed_at=executed_at,
            action="noop",
            reason=reason,
            entry_date=datetime(2026, 1, 2),
        )
        with patch(
            "src.trading.regime_leverage_strategy.service.get_latest_snapshot",
            return_value=snap,
        ), patch(
            "src.trading.regime_leverage_strategy.service.get_last_daily_check_at",
            return_value=last_daily_check_at,
        ), patch(
            "src.trading.regime_leverage_strategy.service.insert_snapshot"
        ) as mock_insert, patch(
            "src.trading.regime_leverage_strategy.service.datetime", _fixed_datetime(now_jst)
        ):
            decision = run_regime_leverage_daily_margin_check(port)
        mock_insert.assert_called_once_with(decision)
        return decision

    def test_weekly_row_is_not_used_as_last_daily_check(self):
        # 最新の行が週次の回(JST 7/18 土 07:00、金曜 7/17 の引けの後)でも、
        # 月曜 7/20 07:30 の日次の回は金曜の足を判定する
        port = _daily_port("2026-07-17", {"2026-07-17": 440.0})
        decision = self._run(
            datetime(2026, 7, 20, 7, 30, tzinfo=_JST),
            datetime(2026, 7, 18, 7, 0),
            port,
            reason="weekly_noop",
        )
        self.assertEqual(decision.reason, "initial_stop")

    def test_weekly_row_latest_and_missed_daily_run_checks_missed_bar(self):
        # 前の日次の回は JST 7/17 金 07:30(7/16 の足まで判定)。週次の回は 7/18 土 07:00。
        # 月曜 7/20 07:30 の回が落ち、火曜 7/21 07:30 に動く。金曜 7/17 と月曜 7/20 の
        # 2本が引けており、落ちた日に見るはずだった金曜の安値が損切りの値段を割っている(#933)
        port = _daily_port("2026-07-20", {"2026-07-17": 440.0})
        decision = self._run(
            datetime(2026, 7, 21, 7, 30, tzinfo=_JST),
            datetime(2026, 7, 18, 7, 0),
            port,
            reason="weekly_noop",
            last_daily_check_at=datetime(2026, 7, 17, 7, 30),
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "initial_stop")
        self.assertEqual(decision.spy_price_usd, 440.0)

    def test_entry_row_latest_and_missed_daily_run_checks_bars_after_entry(self):
        # 週次の回で建てた(JST 7/18 土 07:00、金曜 7/17 の引けで約定)。火曜 7/21 の回が落ち、
        # 水曜 7/22 に動く。建てた後に引けた 7/20 の足の安値が損切りの値段を割っている(#933)
        port = _daily_port("2026-07-21", {"2026-07-20": 440.0})
        decision = self._run(
            datetime(2026, 7, 22, 7, 30, tzinfo=_JST),
            datetime(2026, 7, 18, 7, 0),
            port,
            reason="regime_entry",
            last_daily_check_at=datetime(2026, 7, 18, 7, 0),
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "initial_stop")
        self.assertEqual(decision.spy_price_usd, 440.0)

    def test_summer_us_session_ignores_unclosed_today_bar(self):
        # 夏時間: JST 7/15 23:00 = 米東部 7/15 10:00(場中)。7/15 の足は引けていない
        port = _daily_port("2026-07-15", {"2026-07-15": 300.0})
        decision = self._run(
            datetime(2026, 7, 15, 23, 0, tzinfo=_JST), datetime(2026, 7, 15, 7, 30), port
        )
        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.spy_price_usd, 498.0)

    def test_winter_us_session_ignores_unclosed_today_bar(self):
        # 冬時間: JST 1/14 23:45 = 米東部 1/14 09:45(場中)。1/14 の足は引けていない
        port = _daily_port("2026-01-14", {"2026-01-14": 300.0})
        decision = self._run(
            datetime(2026, 1, 14, 23, 45, tzinfo=_JST), datetime(2026, 1, 14, 7, 30), port
        )
        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.spy_price_usd, 498.0)

    def test_winter_bar_not_closed_until_0600_jst(self):
        # 冬時間: JST 1/15 05:30 = 米東部 1/14 15:30。1/14 の足はまだ引けていない
        port = _daily_port("2026-01-14", {"2026-01-14": 300.0})
        decision = self._run(
            datetime(2026, 1, 15, 5, 30, tzinfo=_JST), datetime(2026, 1, 14, 7, 30), port
        )
        self.assertEqual(decision.action, "noop")

    def test_summer_bar_closed_after_0500_jst(self):
        # 夏時間: JST 7/16 05:30 = 米東部 7/15 16:30。7/15 の足は引けている
        port = _daily_port("2026-07-15", {"2026-07-15": 440.0})
        decision = self._run(
            datetime(2026, 7, 16, 5, 30, tzinfo=_JST), datetime(2026, 7, 15, 7, 30), port
        )
        self.assertEqual(decision.reason, "initial_stop")

    def test_missed_run_checks_older_closed_bar_for_initial_stop(self):
        # 前の回は JST 7/14 07:30(7/13 の足まで判定)。7/15 の回が落ち、7/16 07:30 に動く。
        # 7/14 と 7/15 の2本が引けており、古い方の 7/14 の安値が損切りの値段を割っている
        port = _daily_port("2026-07-15", {"2026-07-14": 440.0})
        decision = self._run(
            datetime(2026, 7, 16, 7, 30, tzinfo=_JST), datetime(2026, 7, 14, 7, 30), port
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "initial_stop")
        self.assertEqual(decision.spy_price_usd, 440.0)

    def test_missed_run_checks_older_closed_bar_for_margin_call(self):
        # 7/14 の安値 300 ドルは維持率 0.20 を割る(マージンコールが損切りより先)
        port = _daily_port("2026-07-15", {"2026-07-14": 300.0})
        decision = self._run(
            datetime(2026, 7, 16, 7, 30, tzinfo=_JST), datetime(2026, 7, 14, 7, 30), port
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "margin_call")
        self.assertEqual(decision.spy_price_usd, 300.0)


if __name__ == "__main__":
    unittest.main()
