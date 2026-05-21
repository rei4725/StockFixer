"""
I-256: 決算・イベントカレンダーフィルター の単体テスト
"""

from __future__ import annotations

import unittest
import unittest.mock
from unittest.mock import MagicMock, patch

import pandas as pd

from src.market_data.event_calendar import (
    _extract_calendar_dates,
    _fetch_event_dates_from_yfinance,
    _load_cached_event_dates,
    _save_event_dates,
    get_event_dates,
    is_near_event,
)
from src.trading.signal_generator import filter_event_signals


# ---------------------------------------------------------------------------
# _extract_calendar_dates
# ---------------------------------------------------------------------------


class TestExtractCalendarDates(unittest.TestCase):
    def test_dict_with_earnings_date_list(self):
        dates: set = set()
        cal = {"Earnings Date": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-04-30")]}
        _extract_calendar_dates(cal, dates)
        self.assertIn(pd.Timestamp("2024-01-31"), dates)
        self.assertIn(pd.Timestamp("2024-04-30"), dates)

    def test_dict_with_single_earnings_date(self):
        dates: set = set()
        cal = {"Earnings Date": pd.Timestamp("2024-02-15")}
        _extract_calendar_dates(cal, dates)
        self.assertIn(pd.Timestamp("2024-02-15"), dates)

    def test_dict_with_earnings_date_lowercase_key(self):
        dates: set = set()
        cal = {"earnings_date": [pd.Timestamp("2024-03-20")]}
        _extract_calendar_dates(cal, dates)
        self.assertIn(pd.Timestamp("2024-03-20"), dates)

    def test_dict_without_earnings_date_key(self):
        dates: set = set()
        cal = {"Revenue": 1000000}
        _extract_calendar_dates(cal, dates)
        self.assertEqual(len(dates), 0)

    def test_dataframe_with_earnings_date_in_index(self):
        dates: set = set()
        cal = pd.DataFrame(
            {"Value": [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-08-01")]},
            index=["Earnings Date", "Other"],
        )
        _extract_calendar_dates(cal, dates)
        self.assertIn(pd.Timestamp("2024-05-01"), dates)

    def test_empty_dict(self):
        dates: set = set()
        _extract_calendar_dates({}, dates)
        self.assertEqual(len(dates), 0)

    def test_invalid_date_skipped(self):
        dates: set = set()
        cal = {"Earnings Date": ["not-a-date", pd.Timestamp("2024-01-01")]}
        _extract_calendar_dates(cal, dates)
        # 有効な日付のみ追加される
        self.assertIn(pd.Timestamp("2024-01-01"), dates)


# ---------------------------------------------------------------------------
# is_near_event
# ---------------------------------------------------------------------------


class TestIsNearEvent(unittest.TestCase):
    def _event_dates(self, *date_strs):
        return pd.DatetimeIndex([pd.Timestamp(d) for d in date_strs])

    def test_exactly_on_event_date(self):
        ev = self._event_dates("2024-02-01")
        self.assertTrue(
            is_near_event("2024-02-01", "us", "AAPL", event_dates=ev)
        )

    def test_two_days_before(self):
        ev = self._event_dates("2024-02-01")
        self.assertTrue(
            is_near_event("2024-01-30", "us", "AAPL", event_dates=ev, days_before=2)
        )

    def test_one_day_after(self):
        ev = self._event_dates("2024-02-01")
        self.assertTrue(
            is_near_event("2024-02-02", "us", "AAPL", event_dates=ev, days_after=1)
        )

    def test_outside_window_before(self):
        ev = self._event_dates("2024-02-01")
        self.assertFalse(
            is_near_event("2024-01-29", "us", "AAPL", event_dates=ev, days_before=2)
        )

    def test_outside_window_after(self):
        ev = self._event_dates("2024-02-01")
        self.assertFalse(
            is_near_event("2024-02-03", "us", "AAPL", event_dates=ev, days_after=1)
        )

    def test_empty_event_dates_returns_false(self):
        self.assertFalse(
            is_near_event("2024-02-01", "us", "AAPL", event_dates=pd.DatetimeIndex([]))
        )

    def test_jp_market(self):
        ev = self._event_dates("2024-05-15")
        self.assertTrue(
            is_near_event("2024-05-14", "jp", "7203", event_dates=ev, days_before=2)
        )


# ---------------------------------------------------------------------------
# filter_event_signals
# ---------------------------------------------------------------------------


class TestFilterEventSignals(unittest.TestCase):
    def _make_signals(self, values, start="2024-01-28"):
        dates = pd.date_range(start=start, periods=len(values), freq="D")
        return pd.Series(values, index=dates)

    def _event_dates(self, *date_strs):
        return pd.DatetimeIndex([pd.Timestamp(d) for d in date_strs])

    def test_buy_near_event_becomes_hold(self):
        # 2024-01-30 は 2024-02-01 の 2日前
        signals = self._make_signals(["Buy", "Buy", "Buy", "Buy", "Buy"], start="2024-01-29")
        ev = self._event_dates("2024-02-01")
        result = filter_event_signals(signals, ev, days_before=2, days_after=1)
        # 2024-01-30 (2日前), 2024-01-31 (1日前), 2024-02-01 (当日), 2024-02-02 (1日後) → Hold
        # 2024-01-29 は window外 (3日前) → Buy のまま
        self.assertEqual(result.iloc[0], "Buy")   # 2024-01-29: 3日前 → Buy
        self.assertEqual(result.iloc[1], "Hold")  # 2024-01-30: 2日前 → Hold
        self.assertEqual(result.iloc[2], "Hold")  # 2024-01-31: 1日前 → Hold
        self.assertEqual(result.iloc[3], "Hold")  # 2024-02-01: 当日 → Hold
        self.assertEqual(result.iloc[4], "Hold")  # 2024-02-02: 1日後 → Hold

    def test_sell_near_event_unchanged(self):
        # Sell シグナルはイベント日近傍でも変換しない
        signals = self._make_signals(["Sell", "Sell", "Sell"], start="2024-01-31")
        ev = self._event_dates("2024-02-01")
        result = filter_event_signals(signals, ev)
        self.assertTrue(all(result == "Sell"))

    def test_empty_event_dates_no_change(self):
        signals = self._make_signals(["Buy", "Buy"])
        result = filter_event_signals(signals, pd.DatetimeIndex([]))
        self.assertTrue(all(result == "Buy"))

    def test_does_not_mutate_original(self):
        signals = self._make_signals(["Buy", "Buy", "Buy"], start="2024-01-31")
        ev = self._event_dates("2024-02-01")
        original = signals.copy()
        filter_event_signals(signals, ev)
        pd.testing.assert_series_equal(signals, original)

    def test_no_buy_signals(self):
        signals = self._make_signals(["Hold", "Hold", "Sell"])
        ev = self._event_dates("2024-01-29")
        result = filter_event_signals(signals, ev)
        pd.testing.assert_series_equal(result, signals)

    def test_buy_outside_window_unchanged(self):
        signals = self._make_signals(["Buy"], start="2024-01-10")
        ev = self._event_dates("2024-02-01")
        result = filter_event_signals(signals, ev)
        self.assertEqual(result.iloc[0], "Buy")


# ---------------------------------------------------------------------------
# get_event_dates (with mocks)
# ---------------------------------------------------------------------------


class TestGetEventDates(unittest.TestCase):
    @patch("src.market_data.event_calendar._load_cached_event_dates")
    def test_returns_cache_when_fresh(self, mock_load):
        expected = pd.DatetimeIndex([pd.Timestamp("2024-02-01")])
        mock_load.return_value = expected
        result = get_event_dates("us", "AAPL")
        pd.testing.assert_index_equal(result, expected)
        mock_load.assert_called_once_with("us", "AAPL")

    @patch("src.market_data.event_calendar._save_event_dates")
    @patch("src.market_data.event_calendar._fetch_event_dates_from_yfinance")
    @patch("src.market_data.event_calendar._load_cached_event_dates")
    def test_fetches_from_yfinance_when_cache_miss(
        self, mock_load, mock_fetch, mock_save
    ):
        mock_load.return_value = None
        fetched = pd.DatetimeIndex([pd.Timestamp("2024-05-01")])
        mock_fetch.return_value = fetched

        result = get_event_dates("jp", "7203")

        mock_fetch.assert_called_once_with("jp", "7203")
        mock_save.assert_called_once_with("jp", "7203", fetched)
        pd.testing.assert_index_equal(result, fetched)

    @patch("src.market_data.event_calendar._save_event_dates")
    @patch("src.market_data.event_calendar._fetch_event_dates_from_yfinance")
    @patch("src.market_data.event_calendar._load_cached_event_dates")
    def test_returns_empty_index_on_fetch_failure(
        self, mock_load, mock_fetch, mock_save
    ):
        mock_load.return_value = None
        mock_fetch.return_value = pd.DatetimeIndex([])

        result = get_event_dates("us", "UNKNOWN")

        self.assertEqual(len(result), 0)
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# _fetch_event_dates_from_yfinance (with yfinance mock)
# ---------------------------------------------------------------------------


class TestFetchEventDatesFromYfinance(unittest.TestCase):
    @patch("src.market_data.event_calendar.yf.Ticker")
    @patch("src.market_data.event_calendar.get_ticker")
    def test_extracts_from_calendar_dict(self, mock_get_ticker, mock_ticker_cls):
        mock_get_ticker.return_value = "AAPL"
        ticker_obj = MagicMock()
        ticker_obj.calendar = {"Earnings Date": [pd.Timestamp("2024-07-25")]}
        ticker_obj.get_earnings_dates.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = ticker_obj

        result = _fetch_event_dates_from_yfinance("us", "AAPL")

        self.assertIn(pd.Timestamp("2024-07-25"), result)

    @patch("src.market_data.event_calendar.yf.Ticker")
    @patch("src.market_data.event_calendar.get_ticker")
    def test_returns_empty_on_exception(self, mock_get_ticker, mock_ticker_cls):
        mock_get_ticker.return_value = "AAPL"
        mock_ticker_cls.side_effect = Exception("network error")

        result = _fetch_event_dates_from_yfinance("us", "AAPL")

        self.assertEqual(len(result), 0)

    @patch("src.market_data.event_calendar.yf.Ticker")
    @patch("src.market_data.event_calendar.get_ticker")
    def test_jp_ticker_uses_dot_t_suffix(self, mock_get_ticker, mock_ticker_cls):
        mock_get_ticker.return_value = "7203.T"
        ticker_obj = MagicMock()
        ticker_obj.calendar = None
        ticker_obj.get_earnings_dates.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = ticker_obj

        _fetch_event_dates_from_yfinance("jp", "7203")

        mock_get_ticker.assert_called_once_with("jp", "7203")
        mock_ticker_cls.assert_called_once_with("7203.T")

    @patch("src.market_data.event_calendar.yf.Ticker")
    @patch("src.market_data.event_calendar.get_ticker")
    def test_deduplicates_dates(self, mock_get_ticker, mock_ticker_cls):
        mock_get_ticker.return_value = "AAPL"
        ticker_obj = MagicMock()
        # calendar と earnings_dates で同じ日付が重複
        ticker_obj.calendar = {"Earnings Date": [pd.Timestamp("2024-07-25")]}
        earnings_df = pd.DataFrame(
            index=pd.DatetimeIndex([pd.Timestamp("2024-07-25"), pd.Timestamp("2024-04-25")])
        )
        ticker_obj.get_earnings_dates.return_value = earnings_df
        mock_ticker_cls.return_value = ticker_obj

        result = _fetch_event_dates_from_yfinance("us", "AAPL")

        # 重複を除去し2件のみ
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# _load_cached_event_dates (with DB mock)
# ---------------------------------------------------------------------------


class TestLoadCachedEventDates(unittest.TestCase):
    @patch("src.market_data.event_calendar._db_connection")
    def test_returns_none_when_no_rows(self, mock_db):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = []
        mock_db.return_value.__enter__.return_value = mock_con
        mock_db.return_value.__exit__.return_value = None

        result = _load_cached_event_dates("us", "AAPL")

        self.assertIsNone(result)

    @patch("src.market_data.event_calendar._db_connection")
    def test_returns_none_when_ttl_expired(self, mock_db):
        old_ts = (pd.Timestamp.now() - pd.Timedelta(days=8)).strftime("%Y-%m-%d %H:%M:%S")
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [("2024-02-01", old_ts)]
        mock_db.return_value.__enter__.return_value = mock_con
        mock_db.return_value.__exit__.return_value = None

        result = _load_cached_event_dates("us", "AAPL")

        self.assertIsNone(result)

    @patch("src.market_data.event_calendar._db_connection")
    def test_returns_dates_when_cache_fresh(self, mock_db):
        recent_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchall.return_value = [
            ("2024-02-01", recent_ts),
            ("2024-05-01", recent_ts),
        ]
        mock_db.return_value.__enter__.return_value = mock_con
        mock_db.return_value.__exit__.return_value = None

        result = _load_cached_event_dates("us", "AAPL")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    @patch("src.market_data.event_calendar._db_connection")
    def test_returns_none_on_db_exception(self, mock_db):
        mock_db.side_effect = Exception("DB error")

        result = _load_cached_event_dates("us", "AAPL")

        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _save_event_dates (with DB mock)
# ---------------------------------------------------------------------------


class TestSaveEventDates(unittest.TestCase):
    @patch("src.market_data.event_calendar._db_connection")
    def test_skips_empty_dates(self, mock_db):
        _save_event_dates("us", "AAPL", pd.DatetimeIndex([]))
        mock_db.assert_not_called()

    @patch("src.market_data.event_calendar._db_connection")
    def test_inserts_dates_into_db(self, mock_db):
        mock_con = MagicMock()
        mock_db.return_value.__enter__.return_value = mock_con
        mock_db.return_value.__exit__.return_value = None

        dates = pd.DatetimeIndex([pd.Timestamp("2024-02-01"), pd.Timestamp("2024-05-01")])
        _save_event_dates("us", "AAPL", dates)

        mock_con.register.assert_called_once()
        mock_con.execute.assert_called_once()

    @patch("src.market_data.event_calendar._db_connection")
    def test_handles_db_exception_gracefully(self, mock_db):
        mock_db.side_effect = Exception("DB error")

        dates = pd.DatetimeIndex([pd.Timestamp("2024-02-01")])
        _save_event_dates("us", "AAPL", dates)


if __name__ == "__main__":
    unittest.main()
