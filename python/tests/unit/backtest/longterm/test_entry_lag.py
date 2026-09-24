"""エントリー約定ラグ。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.execution import ExecutionModel
from src.backtest.longterm import engine
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.engine import resolve_entry_date
from src.backtest.longterm.portfolio import Portfolio
from src.backtest.screening_port import set_backtest_screening_port
from src.domain.types import PositionEvent, TrendCandidate


class _StubScreeningPort:
    """engine が呼ぶ screening ポートの差し替え用スタブ。

    engine はポート経由で screening を呼ぶようになったため、従来の
    `patch.object(engine, "screen_trend_candidates"/"simulate_position", ...)`
    の代わりにこれを注入する。`None` を渡した側は screening BC の実装へ委譲し、
    偽装したい側だけを関数で差し替える（従来の patch と同じ範囲を保つ）。
    """

    def __init__(self, screen_fn=None, simulate_fn=None):
        self._screen_fn = screen_fn
        self._simulate_fn = simulate_fn

    def screen_trend_candidates(self, market, top_n, as_of):
        if self._screen_fn is not None:
            return self._screen_fn(market=market, top_n=top_n, as_of=as_of)
        from src.screening.trend_screener import screen_trend_candidates

        return screen_trend_candidates(market=market, top_n=top_n, as_of=as_of)

    def simulate_position(self, prices, entry_date, rules):
        if self._simulate_fn is not None:
            return self._simulate_fn(prices, entry_date=entry_date, rules=rules)
        from src.screening.hold_engine import simulate_position

        return simulate_position(prices, entry_date=entry_date, rules=rules)


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

    def test_negative_lag_raises_value_error(self):
        """calendar[i + lag] が負インデックスとして末尾から silently 解決される

        事故（例: i=0, lag=-1 で calendar[-1] = カレンダー最終日を返してしまう）
        を防ぐため、負の lag は明示的に拒否する。
        """
        with self.assertRaises(ValueError):
            resolve_entry_date(self.cal, "2024-01-02", -1)

    def test_negative_lag_at_index_zero_would_otherwise_wrap_to_last_day(self):
        """回帰確認: 修正前は screen_date が calendar[0] のとき、

        `calendar[0 + (-1)]` = `calendar[-1]`（カレンダー最終日）という
        誤ったラップアラウンドが起き得た経路。ここでも ValueError を要求する。
        """
        with self.assertRaises(ValueError):
            resolve_entry_date(self.cal, self.cal[0], -1)


class TestConfigDefault(unittest.TestCase):
    def test_execution_lag_defaults_to_one(self):
        self.assertEqual(LongtermBacktestConfig.build(market="us").execution_lag, 1)

    def test_negative_execution_lag_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            LongtermBacktestConfig.build(market="us", execution_lag=-1)


class TestEnterCandidatesFillsAtGivenDate(unittest.TestCase):
    """`enter_candidates` は与えられた entry_date の終値・起点で約定する。

    候補選定（as_of=screen_date）はもはや `enter_candidates` の責務ではなく
    呼び出し側（`run_longterm_backtest`）が事前に済ませてから候補リストを渡す。
    このテストは、渡された entry_date が価格引き当て・`simulate_position` の
    双方に一貫して使われることを確認する。
    """

    _ENTRY_DATE = "2024-01-03"
    _ENTRY_DAY_PRICE = 110.0

    def test_price_and_simulate_use_entry_date(self):
        price_map = {
            "X": pd.DataFrame(
                {
                    "date": ["2024-01-02", self._ENTRY_DATE],
                    "Close": [100.0, self._ENTRY_DAY_PRICE],
                }
            ),
        }
        candidate = TrendCandidate(
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

        simulate_entry_dates: list = []

        def _simulate(series, entry_date, rules):
            simulate_entry_dates.append(entry_date)
            return [ev]

        set_backtest_screening_port(_StubScreeningPort(simulate_fn=_simulate))
        engine.enter_candidates(cfg, portfolio, self._ENTRY_DATE, [candidate], price_map, execution)

        self.assertEqual(simulate_entry_dates, [self._ENTRY_DATE])
        self.assertIn("X", portfolio.positions)
        pos = portfolio.positions["X"]
        self.assertEqual(pos.entry_date, self._ENTRY_DATE)
        self.assertEqual(pos.entry_price, self._ENTRY_DAY_PRICE)


class TestRunLoopDefersEntryToEntryDate(unittest.TestCase):
    """日次ループ全体での screen_date/entry_date 分離の回帰テスト（Important 1）。

    - screen_trend_candidates の as_of はリスクリーン日（screen_date）で呼ばれる。
    - しかし現金の減算・ポジションの計上は screen_date の時点では一切起きず、
      execution_lag 営業日後の entry_date に達したときだけ起きる。
    screen_date と entry_date の紐付けを入れ替えて配線すると、
    screen_date 時点の equity がすでに初期現金から動いてしまい
    （costs 分だけ減るか、holdings 分だけ増える）1 番目のアサーションが、
    あるいは as_of が entry_date で呼ばれて 2 番目のアサーションが落ちる。
    """

    _SCREEN_DATE = "2024-01-02"
    _ENTRY_DATE = "2024-01-03"
    _AFTER = "2024-01-04"
    _INITIAL_CASH = 1000.0

    def test_equity_untouched_on_screen_date_moves_only_on_entry_date(self):
        calendar_dates = [self._SCREEN_DATE, self._ENTRY_DATE, self._AFTER]
        price_map = {
            "X": pd.DataFrame({"date": calendar_dates, "Close": [100.0, 100.0, 100.0]}),
        }
        candidate = TrendCandidate(
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

        as_of_calls: list = []

        def _screen(market, top_n, as_of):
            as_of_calls.append(as_of)
            return [candidate]

        def _simulate(series, entry_date, rules):
            return [
                PositionEvent(
                    date=self._AFTER,
                    action="exit",
                    price=100.0,
                    held_fraction=0.0,
                    reason="x",
                    multiple=1.0,
                )
            ]

        cfg = LongtermBacktestConfig.build(
            market="us",
            start=self._SCREEN_DATE,
            end=self._AFTER,
            rescreen_freq="monthly",
            max_positions=1,
            execution_lag=1,
            initial_cash=self._INITIAL_CASH,
        )

        set_backtest_screening_port(_StubScreeningPort(screen_fn=_screen, simulate_fn=_simulate))
        with patch.object(engine, "load_price_map", return_value=price_map), patch.object(
            engine, "build_calendar", return_value=calendar_dates
        ), patch.object(
            engine, "make_rescreen_dates", return_value=[self._SCREEN_DATE]
        ), patch.object(
            engine,
            "fetch_benchmark_returns",
            return_value={"ticker": "^GSPC", "total_return": 0.0},
        ):
            equity_df, _metrics, _trades = engine.run_longterm_backtest(cfg)

        # スクリーンはリスクリーン日（screen_date）で呼ばれる。
        self.assertEqual(as_of_calls, [self._SCREEN_DATE])

        equity_by_date = dict(zip(equity_df["date"], equity_df["portfolio_value"]))
        # screen_date の時点ではまだ何も約定していないので現金そのまま
        # （costs 分の減少もポジション評価額の混入もない）。
        self.assertEqual(equity_by_date[self._SCREEN_DATE], self._INITIAL_CASH)
        # entry_date で初めて約定コストの分だけ初期現金を下回る。
        self.assertLess(equity_by_date[self._ENTRY_DATE], self._INITIAL_CASH)


if __name__ == "__main__":
    unittest.main()
