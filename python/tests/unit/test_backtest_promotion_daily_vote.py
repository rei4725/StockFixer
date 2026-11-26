"""backtest_promotion/daily_vote.py の票に数える日の決め方を、関数を直に呼んで確かめる（#2196）。"""

from __future__ import annotations

import pandas as pd

from src.prediction.backtest_promotion.daily_vote import (
    VOTE_SKIP_JP_HOLIDAY,
    VOTE_SKIP_JP_SYMBOLS_SHORT,
    split_vote_days,
    vote_need_symbols,
)
from src.prediction.backtest_promotion.judge import MIN_COVERAGE

# 2026-11-24〜26 は火〜木の営業日、2026-11-21・22 は土日（休場）
BIZ_1 = "2026-11-24"
BIZ_2 = "2026-11-25"
BIZ_3 = "2026-11-26"
SAT = "2026-11-21"
SUN = "2026-11-22"

WATCHLIST = {"1111.T", "2222.T"}


def _table(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["market", "symbol", "date"])


class TestVoteNeedSymbols:
    def test_watchlist_count_is_denominator(self) -> None:
        assert vote_need_symbols(10, {"A", "B", "C"}) == 3 * MIN_COVERAGE

    def test_none_watchlist_uses_universe_jp(self) -> None:
        assert vote_need_symbols(10, None) == 10 * MIN_COVERAGE

    def test_empty_watchlist_uses_universe_jp(self) -> None:
        assert vote_need_symbols(7, set()) == 7 * MIN_COVERAGE


class TestSplitVoteDays:
    def test_symbol_is_upper_cased_before_watchlist_match(self) -> None:
        table = _table([("jp", "1111.t", BIZ_1), ("jp", "2222.t", BIZ_1)])
        need = vote_need_symbols(0, WATCHLIST)

        counted, skipped = split_vote_days(table, [BIZ_1], need, WATCHLIST)

        assert counted == [BIZ_1]
        assert skipped == []

    def test_symbols_outside_watchlist_are_not_counted(self) -> None:
        table = _table([("jp", "1111.T", BIZ_1), ("jp", "9999.T", BIZ_1)])
        need = vote_need_symbols(0, WATCHLIST)

        counted, skipped = split_vote_days(table, [BIZ_1], need, WATCHLIST)

        assert counted == []
        assert skipped == [{"date": BIZ_1, "reason": VOTE_SKIP_JP_SYMBOLS_SHORT, "n_jp_symbols": 1}]

    def test_non_jp_rows_are_not_counted(self) -> None:
        table = _table([("jp", "1111.T", BIZ_1), ("us", "AAPL", BIZ_1)])
        need = vote_need_symbols(2, None)

        counted, skipped = split_vote_days(table, [BIZ_1], need, None)

        assert counted == []
        assert skipped == [{"date": BIZ_1, "reason": VOTE_SKIP_JP_SYMBOLS_SHORT, "n_jp_symbols": 1}]

    def test_two_rows_of_same_symbol_and_day_count_once(self) -> None:
        table = _table([("jp", "1111.T", BIZ_1), ("jp", "1111.T", BIZ_1)])
        need = vote_need_symbols(0, WATCHLIST)

        counted, skipped = split_vote_days(table, [BIZ_1], need, WATCHLIST)

        assert counted == []
        assert skipped[0]["n_jp_symbols"] == 1

    def test_day_without_rows_counts_zero_symbols(self) -> None:
        table = _table([("jp", "1111.T", BIZ_1), ("jp", "2222.T", BIZ_1)])
        need = vote_need_symbols(0, WATCHLIST)

        counted, skipped = split_vote_days(table, [BIZ_1, BIZ_2], need, WATCHLIST)

        assert counted == [BIZ_1]
        assert skipped == [{"date": BIZ_2, "reason": VOTE_SKIP_JP_SYMBOLS_SHORT, "n_jp_symbols": 0}]

    def test_holiday_and_short_business_day_and_date_order(self) -> None:
        table = _table(
            [
                ("jp", "1111.T", SAT),
                ("jp", "1111.T", BIZ_1),
                ("jp", "2222.T", BIZ_1),
                ("jp", "1111.T", BIZ_2),
                ("jp", "1111.T", BIZ_3),
                ("jp", "2222.T", BIZ_3),
            ]
        )
        need = vote_need_symbols(0, WATCHLIST)
        dates = [SAT, SUN, BIZ_1, BIZ_2, BIZ_3]

        counted, skipped = split_vote_days(table, dates, need, None)

        assert counted == [BIZ_1, BIZ_3]
        assert skipped == [
            {"date": SAT, "reason": VOTE_SKIP_JP_HOLIDAY, "n_jp_symbols": 1},
            {"date": SUN, "reason": VOTE_SKIP_JP_HOLIDAY, "n_jp_symbols": 0},
            {"date": BIZ_2, "reason": VOTE_SKIP_JP_SYMBOLS_SHORT, "n_jp_symbols": 1},
        ]
