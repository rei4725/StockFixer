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
        # interest_accrued = 72,500.0 * 27 * 0.030 / 365 * 3 = 482.8767...
        # equity_at_low = 1,000,000.0 + 52,920.0 - 482.8767... - 0.0 = 1,052,437.3288
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
        # interest_accrued = 482.8767...(上と同じ, days_held=3)
        # equity = 1,000,000.0 - 69,390.0 - 482.8767... - 0.0 = 930,127.3288
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


if __name__ == "__main__":
    unittest.main()
