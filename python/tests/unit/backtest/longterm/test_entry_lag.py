"""エントリー約定ラグ。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.execution import ExecutionModel
from src.backtest.longterm import engine
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.engine import resolve_entry_date
from src.backtest.longterm.portfolio import Portfolio
from src.screening.types import PositionEvent, TrendCandidate


class TestResolveEntryDate(unittest.TestCase):
    def setUp(self):
        self.cal = ["2024-01-02", "2024-01-03", "2024-01-04"]

    def test_lag_zero_is_same_day(self):
        self.assertEqual(resolve_entry_date(self.cal, "2024-01-02", 0), "2024-01-02")

    def test_lag_one_is_next_trading_day(self):
        self.assertEqual(resolve_entry_date(self.cal, "2024-01-02", 1), "2024-01-03")

    def test_last_day_with_lag_returns_none(self):
        self.assertIsNone(resolve_entry_date(self.cal, "2024-01-04", 1))

    def test_unknown_date_returns_none(self):
        self.assertIsNone(resolve_entry_date(self.cal, "2023-12-31", 1))


class TestConfigDefault(unittest.TestCase):
    def test_execution_lag_defaults_to_one(self):
        self.assertEqual(LongtermBacktestConfig.build(market="us").execution_lag, 1)


class TestEnterCandidatesUsesEntryDateNotScreenDate(unittest.TestCase):
    """screen_date と entry_date を取り違えると検出できる回帰テスト。

    screen_date と entry_date で価格を変え、かつ screen_trend_candidates の
    as_of を捕捉する。screen_date/entry_date の引数を入れ替えて配線すると、
    - entry_price が screen_date 側の値（100.0）になり price アサーションが落ちる
    - as_of が entry_date（"2024-01-03"）で呼ばれ as_of アサーションが落ちる
    ため、両者を確実に検出できる。
    """

    _SCREEN_DATE = "2024-01-02"
    _ENTRY_DATE = "2024-01-03"
    _SCREEN_DAY_PRICE = 100.0
    _ENTRY_DAY_PRICE = 110.0

    def test_price_and_simulate_use_entry_date_screen_uses_screen_date(self):
        price_map = {
            "X": pd.DataFrame(
                {
                    "date": [self._SCREEN_DATE, self._ENTRY_DATE],
                    "Close": [self._SCREEN_DAY_PRICE, self._ENTRY_DAY_PRICE],
                }
            ),
        }
        candidate = TrendCandidate(
            market="us",
            symbol="X",
            score=1.0,
            close=self._SCREEN_DAY_PRICE,
            dist_from_52w_high=0.0,
            above_200dma=True,
            sma200_rising=True,
            return_6m=0.1,
            return_12m=0.2,
            avg_volume=1_000_000.0,
        )
        ev = PositionEvent(
            date=self._ENTRY_DATE,
            action="exit",
            price=self._ENTRY_DAY_PRICE,
            held_fraction=0.0,
            reason="x",
            multiple=1.0,
        )

        cfg = LongtermBacktestConfig.build(market="us", max_positions=1)
        execution = ExecutionModel(cfg.costs)
        portfolio = Portfolio(cash=1000.0)

        as_of_calls: list = []
        simulate_entry_dates: list = []

        def _screen(market, top_n, as_of, **kwargs):
            as_of_calls.append(as_of)
            return [candidate]

        def _simulate(series, entry_date, rules):
            simulate_entry_dates.append(entry_date)
            return [ev]

        with patch.object(engine, "screen_trend_candidates", side_effect=_screen), patch.object(
            engine, "simulate_position", side_effect=_simulate
        ):
            engine.enter_candidates(
                cfg,
                portfolio,
                self._SCREEN_DATE,
                self._ENTRY_DATE,
                price_map,
                execution,
            )

        # スクリーンはスクリーン日で呼ばれる。
        self.assertEqual(as_of_calls, [self._SCREEN_DATE])
        # 約定価格・simulate_position の起点は約定日。
        self.assertEqual(simulate_entry_dates, [self._ENTRY_DATE])
        self.assertIn("X", portfolio.positions)
        pos = portfolio.positions["X"]
        self.assertEqual(pos.entry_date, self._ENTRY_DATE)
        self.assertEqual(pos.entry_price, self._ENTRY_DAY_PRICE)


if __name__ == "__main__":
    unittest.main()
