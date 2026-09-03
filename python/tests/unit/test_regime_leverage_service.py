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


if __name__ == "__main__":
    unittest.main()
