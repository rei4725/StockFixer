"""撤退と部分利確の約定ラグ（#977）。

`simulate_position` は判定した日の終値でイベントを返す。エンジンはその約定を
execution_lag 営業日後の終値へずらし、枠と現金は約定日に空ける。
"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.longterm import engine
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.screening_port import set_backtest_screening_port
from src.domain.types import PositionEvent, TrendCandidate

_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]
_CLOSES = {
    "X": [100.0, 100.0, 200.0, 150.0, 120.0, 110.0],
    "Y": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
}


def _candidate(symbol: str) -> TrendCandidate:
    return TrendCandidate(
        market="us",
        symbol=symbol,
        score=1.0,
        close=100.0,
        dist_from_52w_high=0.0,
        above_200dma=True,
        sma200_rising=True,
        return_6m=0.1,
        return_12m=0.2,
        avg_volume=1_000_000.0,
    )


def _event(symbol: str, i: int, action: str, held_fraction: float, reason: str) -> PositionEvent:
    """hold_engine と同じく、判定した日の終値を price に入れたイベントを作る。"""
    price = _CLOSES[symbol][i]
    return PositionEvent(
        date=_DATES[i],
        action=action,
        price=price,
        held_fraction=held_fraction,
        reason=reason,
        multiple=price / 100.0,
    )


class _StubScreeningPort:
    def __init__(self, screens: dict, events: dict):
        self._screens = screens
        self._events = events

    def screen_trend_candidates(self, market, top_n, as_of):
        return [_candidate(s) for s in self._screens.get(as_of, [])]

    def simulate_position(self, prices, entry_date, rules):
        symbol = "X" if prices["Close"].tolist() == _CLOSES["X"] else "Y"
        return list(self._events[symbol])


def _run(lag: int, screens: dict, events: dict):
    price_map = {s: pd.DataFrame({"date": _DATES, "Close": c}) for s, c in _CLOSES.items()}
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
    )
    set_backtest_screening_port(_StubScreeningPort(screens, events))
    with patch.object(engine, "load_price_map", return_value=price_map), patch.object(
        engine, "make_rescreen_dates", return_value=sorted(screens)
    ), patch.object(
        engine,
        "fetch_benchmark_returns",
        return_value={"ticker": "^GSPC", "total_return": 0.0},
    ):
        return engine.run_longterm_backtest(cfg)


# X は 2 本目の判定で半分を利確（scale_2x）し、3 本目の判定でトレーリングで撤退する。
_X_EVENTS = [
    _event("X", 2, "scale_out", 0.5, "scale_2x"),
    _event("X", 3, "exit", 0.0, "trail_stop"),
]


class TestExitLag(unittest.TestCase):
    def test_lag_one_fills_scale_out_and_exit_next_day(self):
        """判定の翌営業日の終値で売り代金・exit_date・exit_price・multiple を出す。"""
        equity, _, trades = _run(1, {_DATES[0]: ["X"]}, {"X": _X_EVENTS})

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(trade["entry_date"], _DATES[1])
        self.assertEqual(trade["exit_date"], _DATES[4])
        self.assertAlmostEqual(trade["exit_price"], 120.0)
        self.assertAlmostEqual(trade["multiple"], 1.2)
        self.assertEqual(trade["exit_reason"], "trail_stop")
        # 10 株を 100 で買い、5 株を 150（scale_out の翌日）、5 株を 120 で売る。
        final = dict(zip(equity["date"], equity["portfolio_value"]))
        self.assertAlmostEqual(final[_DATES[-1]], 5 * 150.0 + 5 * 120.0)

    def test_lag_one_scale_out_cash_arrives_on_fill_day(self):
        """部分利確の判定の日は、まだ全株を保有として評価する。"""
        equity, _, _ = _run(1, {_DATES[0]: ["X"]}, {"X": _X_EVENTS})
        by_date = dict(zip(equity["date"], equity["portfolio_value"]))
        # 判定の日（終値 200）は 10 株とも保有のまま。
        self.assertAlmostEqual(by_date[_DATES[2]], 10 * 200.0)
        # 約定の日（終値 150）に 5 株を 150 で売り、残り 5 株を 150 で評価する。
        self.assertAlmostEqual(by_date[_DATES[3]], 10 * 150.0)

    def test_lag_zero_keeps_same_day_fill(self):
        """execution_lag=0 なら判定した日の終値で約定する（今までと同じ）。"""
        equity, _, trades = _run(0, {_DATES[0]: ["X"]}, {"X": _X_EVENTS})

        trade = trades.iloc[0]
        self.assertEqual(trade["entry_date"], _DATES[0])
        self.assertEqual(trade["exit_date"], _DATES[3])
        self.assertAlmostEqual(trade["exit_price"], 150.0)
        self.assertAlmostEqual(trade["multiple"], 1.5)
        final = dict(zip(equity["date"], equity["portfolio_value"]))
        self.assertAlmostEqual(final[_DATES[-1]], 5 * 200.0 + 5 * 150.0)

    def test_exit_judged_on_last_day_fills_at_last_close(self):
        """判定の日がカレンダーの末尾なら、末尾の日の終値で手仕舞う。"""
        events = {"X": [_event("X", 5, "exit", 0.0, "trail_stop")]}
        _, _, trades = _run(1, {_DATES[0]: ["X"]}, events)

        self.assertEqual(len(trades), 1)
        trade = trades.iloc[0]
        self.assertEqual(trade["exit_date"], _DATES[5])
        self.assertAlmostEqual(trade["exit_price"], 110.0)
        self.assertAlmostEqual(trade["multiple"], 1.1)

    def test_slot_stays_filled_until_exit_fill_day(self):
        """撤退の約定日までは枠が埋まったままで、同じ日の入りに枠が空かない。"""
        # X は 2 本目で撤退を判定（約定は 3 本目）。Y は 1 本目のスクリーンで 2 本目に入ろうとする。
        events = {
            "X": [_event("X", 2, "exit", 0.0, "trail_stop")],
            "Y": [_event("Y", 5, "exit", 0.0, "end_of_data")],
        }
        equity, _, trades = _run(1, {_DATES[0]: ["X"], _DATES[1]: ["Y"]}, events)

        self.assertEqual(list(trades["symbol"]), ["X"])
        self.assertEqual(trades.iloc[0]["exit_date"], _DATES[3])
        by_date = dict(zip(equity["date"], equity["portfolio_value"]))
        # 判定の日（終値 200）は X を 10 株保有のまま評価する。
        self.assertAlmostEqual(by_date[_DATES[2]], 10 * 200.0)


if __name__ == "__main__":
    unittest.main()
