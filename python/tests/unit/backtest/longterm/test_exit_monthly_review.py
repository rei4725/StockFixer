"""撤退と部分利確の判定を本番の月次の判定日に丸める（#1488）。

本番の multibagger は毎月1日にその前の月の終値までで保有を判定して知らせ、
人間はその後に売る。exit_review_freq="monthly" のとき、エンジンは判定の日を
その月の最後の取引日に丸めてから execution_lag 営業日後の終値で約定させる。
既定の "daily" は丸めない（#977 の動きのまま）。
"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.longterm import engine
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.screening_port import set_backtest_screening_port
from src.domain.types import PositionEvent, TrendCandidate

_DATES = ["2024-01-26", "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]
_CLOSES = [100.0, 100.0, 200.0, 150.0, 120.0, 110.0]


def _candidate() -> TrendCandidate:
    return TrendCandidate(
        market="us",
        symbol="X",
        score=1.0,
        close=100.0,
        dist_from_52w_high=0.0,
        above_200dma=True,
        sma200_rising=True,
        return_6m=0.1,
        return_12m=0.2,
        avg_volume=1_000_000.0,
    )


def _event(i: int, action: str, held_fraction: float, reason: str) -> PositionEvent:
    """hold_engine と同じく、判定した日の終値を price に入れたイベントを作る。"""
    return PositionEvent(
        date=_DATES[i],
        action=action,
        price=_CLOSES[i],
        held_fraction=held_fraction,
        reason=reason,
        multiple=_CLOSES[i] / 100.0,
    )


class _StubScreeningPort:
    def __init__(self, events: list):
        self._events = events

    def screen_trend_candidates(self, market, top_n, as_of):
        return [_candidate()]

    def simulate_position(self, prices, entry_date, rules):
        return list(self._events)


def _run(lag: int, events: list, **kwargs):
    price_map = {"X": pd.DataFrame({"date": _DATES, "Close": _CLOSES})}
    cfg = LongtermBacktestConfig.build(
        market="us",
        fee_rate=0.0,
        slippage=0.0,
        start=_DATES[0],
        end=_DATES[-1],
        rescreen_freq="monthly",
        max_positions=1,
        execution_lag=lag,
        initial_cash=1000.0,
        dynamic_slippage=False,
        **kwargs,
    )
    set_backtest_screening_port(_StubScreeningPort(events))
    with patch.object(engine, "load_price_map", return_value=price_map), patch.object(
        engine, "make_rescreen_dates", return_value=[_DATES[0]]
    ), patch.object(
        engine,
        "fetch_benchmark_returns",
        return_value={"ticker": "^GSPC", "total_return": 0.0},
    ):
        return engine.run_longterm_backtest(cfg)


class TestExitMonthlyReview(unittest.TestCase):
    def test_default_is_daily(self):
        self.assertEqual(LongtermBacktestConfig.build(market="us").exit_review_freq, "daily")

    def test_unknown_freq_rejected(self):
        with self.assertRaises(ValueError):
            LongtermBacktestConfig.build(market="us", exit_review_freq="weekly")

    def test_monthly_mid_month_exit_fills_after_month_end(self):
        """月の途中（1/30）の撤退は、月末 1/31 の lag 営業日後（2/1）の終値で約定する。"""
        events = [_event(2, "exit", 0.0, "trail_stop")]
        equity, _, trades = _run(1, events, exit_review_freq="monthly")

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(trade["exit_date"], _DATES[4])
        self.assertAlmostEqual(trade["exit_price"], 120.0)
        self.assertAlmostEqual(trade["multiple"], 1.2)
        by_date = dict(zip(equity["date"], equity["portfolio_value"]))
        # 月末（終値 150）はまだ 10 株を保有のまま評価する。
        self.assertAlmostEqual(by_date[_DATES[3]], 10 * 150.0)
        self.assertAlmostEqual(by_date[_DATES[-1]], 10 * 120.0)

    def test_daily_mid_month_exit_fills_next_day(self):
        """既定の daily は丸めず、判定の翌営業日（1/31）の終値で約定する（#977 のまま）。"""
        events = [_event(2, "exit", 0.0, "trail_stop")]
        _, _, trades = _run(1, events)

        trade = trades.iloc[0]
        self.assertEqual(trade["exit_date"], _DATES[3])
        self.assertAlmostEqual(trade["exit_price"], 150.0)

    def test_monthly_scale_out_is_rounded_too(self):
        """部分利確も撤退と同じく月末に丸めてから lag 営業日後に約定する。"""
        events = [
            _event(2, "scale_out", 0.5, "scale_2x"),
            _event(3, "exit", 0.0, "trail_stop"),
        ]
        equity, _, trades = _run(1, events, exit_review_freq="monthly")

        trade = trades.iloc[0]
        self.assertEqual(trade["exit_date"], _DATES[4])
        self.assertAlmostEqual(trade["exit_price"], 120.0)
        # 10 株を 100 で買い、5 株ずつ 2/1 の終値 120 で売る。
        final = dict(zip(equity["date"], equity["portfolio_value"]))
        self.assertAlmostEqual(final[_DATES[-1]], 10 * 120.0)

    def test_monthly_with_lag_zero_fills_on_month_end(self):
        """lag=0 でも月次なら月末の終値で約定する。"""
        events = [_event(2, "exit", 0.0, "trail_stop")]
        _, _, trades = _run(0, events, exit_review_freq="monthly")

        trade = trades.iloc[0]
        self.assertEqual(trade["exit_date"], _DATES[3])
        self.assertAlmostEqual(trade["exit_price"], 150.0)


if __name__ == "__main__":
    unittest.main()
