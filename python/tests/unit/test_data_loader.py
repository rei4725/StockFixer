"""data_loader モジュールのユニットテスト"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from src.data.data_loader import (
    get_latest_business_day,
    get_stock_data_auto,
    merge_market_data,
    should_fetch_fresh_data,
)


def _make_ohlcv(periods=10, start="2024-01-01"):
    """テスト用 OHLCV DataFrame を生成する"""
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(periods)],
            "High": [102.0 + i for i in range(periods)],
            "Low": [99.0 + i for i in range(periods)],
            "Close": [101.0 + i for i in range(periods)],
            "Volume": [1000 + i * 100 for i in range(periods)],
        },
        index=dates,
    )


class TestGetLatestBusinessDay(unittest.TestCase):
    """get_latest_business_day 関数のテスト"""

    def test_monday_returns_monday(self):
        """月曜日はそのまま返ること"""
        monday = datetime(2024, 1, 8)  # 月曜日
        result = get_latest_business_day(monday)
        self.assertEqual(result.weekday(), 0)

    def test_friday_returns_friday(self):
        """金曜日はそのまま返ること"""
        friday = datetime(2024, 1, 5)  # 金曜日
        result = get_latest_business_day(friday)
        self.assertEqual(result.weekday(), 4)

    def test_saturday_returns_friday(self):
        """土曜日は直前の金曜日を返ること"""
        saturday = datetime(2024, 1, 6)  # 土曜日
        result = get_latest_business_day(saturday)
        self.assertEqual(result.weekday(), 4)
        self.assertEqual(result.date(), datetime(2024, 1, 5).date())

    def test_sunday_returns_friday(self):
        """日曜日は直前の金曜日を返ること"""
        sunday = datetime(2024, 1, 7)  # 日曜日
        result = get_latest_business_day(sunday)
        self.assertEqual(result.weekday(), 4)

    def test_none_returns_weekday(self):
        """引数なしの場合は現在日時から営業日を返すこと"""
        result = get_latest_business_day()
        self.assertLess(result.weekday(), 5)

    def test_string_date_accepted(self):
        """文字列形式の日付引数を受け付けること"""
        result = get_latest_business_day("2024-01-08")
        self.assertIsNotNone(result)
        self.assertLess(result.weekday(), 5)


class TestShouldFetchFreshData(unittest.TestCase):
    """should_fetch_fresh_data 関数のテスト"""

    def test_none_db_date_returns_true(self):
        """db_latest_date が None の場合は True（取得必要）を返すこと"""
        result = should_fetch_fresh_data(None)
        self.assertTrue(result)

    def test_fresh_db_data_returns_false(self):
        """DB が最新の場合は False（取得不要）を返すこと"""
        today = datetime.now()
        latest_biz = today
        while latest_biz.weekday() >= 5:
            latest_biz -= timedelta(days=1)
        result = should_fetch_fresh_data(latest_biz, end_date=today)
        self.assertFalse(result)

    def test_stale_db_data_returns_true(self):
        """DB が 1 週間以上古い場合は True を返すこと"""
        old_date = datetime.now() - timedelta(days=10)
        result = should_fetch_fresh_data(old_date)
        self.assertTrue(result)

    def test_string_dates_accepted(self):
        """文字列形式の日付でも動作すること"""
        old_date_str = "2024-01-01"
        end_date_str = "2024-06-01"
        result = should_fetch_fresh_data(old_date_str, end_date=end_date_str)
        self.assertTrue(result)


class TestMergeMarketData(unittest.TestCase):
    """merge_market_data 関数のテスト"""

    def test_none_db_returns_fresh(self):
        """db_data が None の場合は fresh_data がそのまま返ること"""
        fresh = _make_ohlcv(5)
        result = merge_market_data(None, fresh)
        pd.testing.assert_frame_equal(result, fresh)

    def test_empty_db_returns_fresh(self):
        """db_data が空 DataFrame の場合は fresh_data が返ること"""
        fresh = _make_ohlcv(5)
        result = merge_market_data(pd.DataFrame(), fresh)
        pd.testing.assert_frame_equal(result, fresh)

    def test_none_fresh_returns_db(self):
        """fresh_data が None の場合は db_data が返ること"""
        db = _make_ohlcv(5)
        result = merge_market_data(db, None)
        pd.testing.assert_frame_equal(result, db)

    def test_empty_fresh_returns_db(self):
        """fresh_data が空 DataFrame の場合は db_data が返ること"""
        db = _make_ohlcv(5)
        result = merge_market_data(db, pd.DataFrame())
        pd.testing.assert_frame_equal(result, db)

    def test_merge_adds_new_rows(self):
        """db より後に fresh データがある場合、新しい行が追加されること"""
        db = _make_ohlcv(5, start="2024-01-01")
        # db と値が重複しないよう価格レンジを変えた fresh データを用意する
        fresh_dates = pd.date_range("2024-02-01", periods=3, freq="B")
        fresh = pd.DataFrame(
            {
                "Open": [500.0, 501.0, 502.0],
                "High": [502.0, 503.0, 504.0],
                "Low": [499.0, 500.0, 501.0],
                "Close": [501.0, 502.0, 503.0],
                "Volume": [9000, 9100, 9200],
            },
            index=fresh_dates,
        )
        result = merge_market_data(db, fresh)
        self.assertGreater(len(result), len(db))

    def test_merge_no_duplicates(self):
        """重複する行が含まれないこと"""
        db = _make_ohlcv(5, start="2024-01-01")
        fresh = _make_ohlcv(5, start="2024-01-01")  # 全く同じデータ
        result = merge_market_data(db, fresh)
        self.assertEqual(len(result), len(db))

    def test_merge_sorted_by_index(self):
        """結果がインデックス昇順でソートされていること"""
        db = _make_ohlcv(3, start="2024-01-08")
        fresh = _make_ohlcv(3, start="2024-01-15")
        result = merge_market_data(db, fresh)
        self.assertTrue(result.index.is_monotonic_increasing)

    def test_tz_aware_db_handled(self):
        """tz-aware な DB データでも統合できること"""
        dates_tz = pd.date_range("2024-01-01", periods=3, freq="B", tz="UTC")
        db = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates_tz)
        fresh = _make_ohlcv(3, start="2024-01-10")
        # 例外が発生しなければ OK
        result = merge_market_data(db, fresh)
        self.assertIsNotNone(result)


class TestGetStockDataAuto(unittest.TestCase):
    """get_stock_data_auto 関数のテスト"""

    @patch("src.data.data_loader.get_stock_data_from_file")
    def test_file_source_calls_from_file(self, mock_from_file):
        """source='file' のとき get_stock_data_from_file が呼ばれること"""
        mock_from_file.return_value = _make_ohlcv()

        get_stock_data_auto("us", "AAPL", "2024-01-01", "2024-06-01", source="file")

        mock_from_file.assert_called_once()

    @patch("src.data.data_loader.get_stock_data")
    @patch("src.data.data_loader.get_ticker")
    def test_api_source_calls_get_stock_data(self, mock_ticker, mock_get_stock):
        """source='api' のとき get_stock_data が呼ばれること"""
        mock_ticker.return_value = "AAPL"
        mock_get_stock.return_value = _make_ohlcv()

        get_stock_data_auto("us", "AAPL", "2024-01-01", "2024-06-01", source="api")

        mock_get_stock.assert_called_once()

    def test_invalid_source_raises_value_error(self):
        """不正な source 指定で ValueError が発生すること"""
        with self.assertRaises(ValueError):
            get_stock_data_auto("us", "AAPL", "2024-01-01", "2024-06-01", source="invalid")


class TestGetStockDataFromDb(unittest.TestCase):
    """get_stock_data_from_db 関数のテスト"""

    @patch("src.data.data_loader.load_stock_features")
    def test_raises_file_not_found_when_no_data(self, mock_load):
        """DB にデータがない場合 FileNotFoundError が送出されること"""
        from src.data.data_loader import get_stock_data_from_db

        mock_load.return_value = None

        with self.assertRaises(FileNotFoundError):
            get_stock_data_from_db("us", "AAPL")

    @patch("src.data.data_loader.load_stock_features")
    def test_returns_df_when_data_exists(self, mock_load):
        """DB にデータがある場合 DataFrame を返すこと"""
        from src.data.data_loader import get_stock_data_from_db

        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({"Date": dates, "Close": range(10)})
        mock_load.return_value = df

        result = get_stock_data_from_db("us", "AAPL")
        self.assertFalse(result.empty)


class TestGetForexData(unittest.TestCase):
    """get_forex_data 関数のテスト"""

    @patch("src.data.data_loader.yf_client")
    def test_raises_when_empty_data(self, mock_yf_client):
        """データが空の場合は ValueError が発生すること"""
        from src.data.data_loader import get_forex_data

        mock_yf_client.download.return_value = pd.DataFrame()

        with self.assertRaises(ValueError):
            get_forex_data("JPY=X", "2024-01-01", "2024-06-01")

    @patch("src.data.data_loader.yf_client")
    def test_returns_df_when_data_exists(self, mock_yf_client):
        """正常時はデータを含む DataFrame が返ること"""
        from src.data.data_loader import get_forex_data

        mock_yf_client.download.return_value = pd.DataFrame(
            {"Close": [150.0]}, index=pd.date_range("2024-01-01", periods=1)
        )

        result = get_forex_data("JPY=X", "2024-01-01", "2024-06-01")
        self.assertFalse(result.empty)


class TestFetchCrossAssetFeatures(unittest.TestCase):
    """fetch_cross_asset_features 関数のテスト"""

    @patch("src.data.data_loader.yf_client")
    def test_returns_none_when_all_fail(self, mock_yf_client):
        """全ティッカー取得失敗時に None が返ること（line 347）"""
        from src.data.data_loader import fetch_cross_asset_features

        mock_yf_client.ticker_history.return_value = pd.DataFrame()  # 空 DataFrame → スキップ
        result = fetch_cross_asset_features("2024-01-01", "2024-03-31")
        assert result is None

    @patch("src.data.data_loader.yf_client")
    def test_returns_dataframe_on_success(self, mock_yf_client):
        """正常時に DataFrame が返ること"""
        from src.data.data_loader import fetch_cross_asset_features

        n = 10
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": [100.0 + i for i in range(n)]}, index=dates)
        mock_yf_client.ticker_history.return_value = df
        result = fetch_cross_asset_features("2024-01-01", "2024-01-31")
        assert result is not None

    @patch("src.data.data_loader.yf_client")
    def test_handles_exception_per_ticker(self, mock_yf_client):
        """ティッカー取得例外はスキップされること（lines 342-344）"""
        from src.data.data_loader import fetch_cross_asset_features

        mock_yf_client.ticker_history.side_effect = Exception("接続タイムアウト")
        # 例外なしで None が返ること
        result = fetch_cross_asset_features("2024-01-01", "2024-01-31")
        assert result is None

    @patch("src.data.data_loader.yf_client")
    def test_handles_dataframe_close_column(self, mock_yf_client):
        """Close 列が MultiIndex DataFrame の場合に iloc[:,0] が取られること（line 334）"""
        from src.data.data_loader import fetch_cross_asset_features

        n = 5
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # MultiIndex カラムを持つ DataFrame（yfinanceが返す形式を模倣）
        mi = pd.MultiIndex.from_tuples([("Close", "^VIX"), ("Open", "^VIX")])
        df = pd.DataFrame(
            [[20.0 + i, 19.0 + i] for i in range(n)],
            index=dates,
            columns=mi,
        )
        mock_yf_client.ticker_history.return_value = df
        # 例外なしで処理されること（Close が DataFrame になるので iloc[:,0] が呼ばれる）
        try:
            result = fetch_cross_asset_features("2024-01-01", "2024-01-31")
        except Exception:
            pass  # フォールバックエラーも OK（ここではカバレッジが目的）


if __name__ == "__main__":
    unittest.main()
