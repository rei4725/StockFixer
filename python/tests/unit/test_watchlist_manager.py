"""Unit tests for watchlist_manager.py"""

import unittest
from unittest.mock import patch

import pandas as pd


class TestFetchSp500Symbols(unittest.TestCase):
    """fetch_sp500_symbols のテスト"""

    @patch("pandas.read_html")
    def test_returns_list_of_symbols(self, mock_html):
        """Wikipedia から S&P500 銘柄リストが返ること"""
        from src.services.watchlist.watchlist_manager import fetch_sp500_symbols

        mock_html.return_value = [pd.DataFrame({"Symbol": ["AAPL", "MSFT", "GOOG"]})]
        result = fetch_sp500_symbols()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    @patch("pandas.read_html")
    def test_returns_empty_on_error(self, mock_html):
        """取得エラー時は空リストが返ること"""
        from src.services.watchlist.watchlist_manager import fetch_sp500_symbols

        mock_html.side_effect = Exception("接続エラー")
        result = fetch_sp500_symbols()
        self.assertEqual(result, [])

    @patch("pandas.read_html")
    def test_dot_replaced_with_dash(self, mock_html):
        """ティッカーの '.' が '-' に置換されること"""
        from src.services.watchlist.watchlist_manager import fetch_sp500_symbols

        mock_html.return_value = [pd.DataFrame({"Symbol": ["BRK.B", "BF.B"]})]
        result = fetch_sp500_symbols()
        self.assertIn("BRK-B", result)
        self.assertIn("BF-B", result)


class TestFetchNikkei225Symbols(unittest.TestCase):
    """fetch_nikkei225_symbols のテスト"""

    @patch("pandas.read_html")
    def test_returns_list_of_symbols(self, mock_html):
        """Wikipedia から日経225 銘柄リストが返ること"""
        from src.services.watchlist.watchlist_manager import fetch_nikkei225_symbols

        # 実際のカラム名は "Code"（英字）
        mock_html.return_value = [pd.DataFrame({"Code": ["7203", "9984", "6758"]})]
        result = fetch_nikkei225_symbols()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    @patch("pandas.read_html")
    def test_returns_empty_on_error(self, mock_html):
        """取得エラー時は空リストが返ること"""
        from src.services.watchlist.watchlist_manager import fetch_nikkei225_symbols

        mock_html.side_effect = Exception("接続エラー")
        result = fetch_nikkei225_symbols()
        self.assertEqual(result, [])

    @patch("pandas.read_html")
    def test_filters_non_4digit_codes(self, mock_html):
        """数字4桁以外のコードは除外されること"""
        from src.services.watchlist.watchlist_manager import fetch_nikkei225_symbols

        mock_html.return_value = [pd.DataFrame({"Code": ["7203", "9984", "ABC", "12345", "0001"]})]
        result = fetch_nikkei225_symbols()
        # "7203", "9984", "0001" の3件のみ残る
        self.assertEqual(len(result), 3)
        self.assertNotIn("ABC", result)
        self.assertNotIn("12345", result)

    @patch("pandas.read_html")
    def test_returns_empty_when_no_code_column(self, mock_html):
        """Code カラムがないテーブルのみの場合は空リストが返ること"""
        from src.services.watchlist.watchlist_manager import fetch_nikkei225_symbols

        mock_html.return_value = [pd.DataFrame({"Symbol": ["AAPL"]})]
        result = fetch_nikkei225_symbols()
        self.assertEqual(result, [])


class TestFetchIndexSymbols(unittest.TestCase):
    """fetch_index_symbols のテスト"""

    @patch("src.services.watchlist.watchlist_manager.fetch_sp500_symbols")
    def test_us_market_calls_sp500(self, mock_sp500):
        """us マーケット指定で fetch_sp500_symbols が呼ばれること"""
        from src.services.watchlist.watchlist_manager import fetch_index_symbols

        mock_sp500.return_value = ["AAPL", "MSFT"]
        result = fetch_index_symbols("us")
        mock_sp500.assert_called_once()
        self.assertEqual(result, ["AAPL", "MSFT"])

    @patch("src.services.watchlist.watchlist_manager.fetch_nikkei225_symbols")
    def test_jp_market_calls_nikkei225(self, mock_nikkei):
        """jp マーケット指定で fetch_nikkei225_symbols が呼ばれること"""
        from src.services.watchlist.watchlist_manager import fetch_index_symbols

        mock_nikkei.return_value = ["7203", "9984"]
        result = fetch_index_symbols("jp")
        mock_nikkei.assert_called_once()
        self.assertEqual(result, ["7203", "9984"])

    def test_unknown_market_returns_empty(self):
        """未知のマーケットは空リストが返ること"""
        from src.services.watchlist.watchlist_manager import fetch_index_symbols

        result = fetch_index_symbols("unknown_market")
        self.assertEqual(result, [])

    @patch("src.services.watchlist.watchlist_manager.fetch_sp500_symbols")
    def test_us_market_case_insensitive(self, mock_sp500):
        """大文字 'US' でも fetch_sp500_symbols が呼ばれること"""
        from src.services.watchlist.watchlist_manager import fetch_index_symbols

        mock_sp500.return_value = ["AAPL"]
        result = fetch_index_symbols("US")
        mock_sp500.assert_called_once()
        self.assertEqual(result, ["AAPL"])


class TestVerifySymbolActive(unittest.TestCase):
    """verify_symbol_active のテスト"""

    @patch("yfinance.Ticker")
    def test_active_symbol_returns_true(self, mock_ticker_cls):
        """有効な銘柄で True が返ること"""
        from src.services.watchlist.watchlist_manager import verify_symbol_active

        mock_ticker = mock_ticker_cls.return_value
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.date_range("2026-01-01", periods=2),
        )
        result = verify_symbol_active("AAPL", "us")
        self.assertTrue(result)

    @patch("yfinance.Ticker")
    def test_inactive_symbol_returns_false(self, mock_ticker_cls):
        """空ヒストリーの銘柄で False が返ること"""
        from src.services.watchlist.watchlist_manager import verify_symbol_active

        mock_ticker = mock_ticker_cls.return_value
        mock_ticker.history.return_value = pd.DataFrame()
        result = verify_symbol_active("INVALID", "us")
        self.assertFalse(result)

    @patch("yfinance.Ticker")
    def test_exception_returns_false(self, mock_ticker_cls):
        """例外発生時は False が返ること"""
        from src.services.watchlist.watchlist_manager import verify_symbol_active

        mock_ticker_cls.side_effect = Exception("APIエラー")
        result = verify_symbol_active("AAPL", "us")
        self.assertFalse(result)

    @patch("yfinance.Ticker")
    def test_jp_symbol_active(self, mock_ticker_cls):
        """日本株 ('jp') でも正常に True が返ること"""
        from src.services.watchlist.watchlist_manager import verify_symbol_active

        mock_ticker = mock_ticker_cls.return_value
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [3000.0]},
            index=pd.date_range("2026-01-01", periods=1),
        )
        result = verify_symbol_active("7203", "jp")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()


class TestDiffWatchlist(unittest.TestCase):
    """diff_watchlist のテスト"""

    @patch("src.services.watchlist.watchlist_manager.verify_symbol_active")
    def test_new_symbols_are_added(self, mock_verify):
        """新規銘柄が added に追加されること"""
        from src.services.watchlist.watchlist_manager import diff_watchlist

        mock_verify.return_value = True
        current = ["AAPL", "MSFT"]
        fetched = ["AAPL", "MSFT", "GOOG"]
        result = diff_watchlist(current, fetched, "us")
        assert "GOOG" in result.added

    @patch("src.services.watchlist.watchlist_manager.verify_symbol_active")
    def test_removed_symbols_go_to_removed_list(self, mock_verify):
        """フェッチされなくなった銘柄が removed または removed_unverified に移ること"""
        from src.services.watchlist.watchlist_manager import diff_watchlist

        mock_verify.return_value = True
        current = ["AAPL", "MSFT", "GOOG"]
        fetched = ["AAPL", "MSFT"]
        result = diff_watchlist(current, fetched, "us")
        assert "GOOG" in result.removed or "GOOG" in result.removed_unverified

    @patch("src.services.watchlist.watchlist_manager.verify_symbol_active")
    def test_unchanged_symbols_are_kept(self, mock_verify):
        """変更なし銘柄が kept に残ること"""
        from src.services.watchlist.watchlist_manager import diff_watchlist

        mock_verify.return_value = True
        current = ["AAPL", "MSFT"]
        fetched = ["AAPL", "MSFT"]
        result = diff_watchlist(current, fetched, "us")
        assert "AAPL" in result.kept
        assert "MSFT" in result.kept

    @patch("src.services.watchlist.watchlist_manager.verify_symbol_active")
    def test_empty_inputs(self, mock_verify):
        """空リスト同士で例外なし"""
        from src.services.watchlist.watchlist_manager import diff_watchlist

        mock_verify.return_value = True
        result = diff_watchlist([], [], "us")
        assert isinstance(result.added, list)
        assert isinstance(result.removed, list)


class TestRunWatchlistRefresh(unittest.TestCase):
    """run_watchlist_refresh のテスト"""

    @patch("src.services.watchlist.watchlist_manager.save_index_membership_snapshot")
    @patch("src.services.watchlist.watchlist_manager.apply_watchlist_update")
    @patch("src.services.watchlist.watchlist_manager.diff_watchlist")
    @patch("src.services.watchlist.watchlist_manager.fetch_index_symbols")
    @patch("src.services.watchlist.watchlist_manager._load_watchlist")
    def test_runs_for_all_markets(
        self, mock_load, mock_fetch, mock_diff, mock_apply, mock_save_snapshot
    ):
        """全マーケットに対して更新フローが実行されること"""
        from src.services.watchlist.watchlist_manager import WatchlistDiff, run_watchlist_refresh

        mock_load.return_value = {"jp": ["7203"], "us": ["AAPL"]}
        mock_fetch.return_value = ["7203", "9984"]
        mock_diff.return_value = WatchlistDiff(
            market="jp",
            added=["9984"],
            kept=["7203"],
            removed=[],
            removed_unverified=[],
            capped=False,
        )
        result = run_watchlist_refresh()
        assert isinstance(result, list)
        mock_apply.assert_called_once()
        self.assertEqual(mock_save_snapshot.call_count, 2)

    @patch("src.services.watchlist.watchlist_manager.save_index_membership_snapshot")
    @patch("src.services.watchlist.watchlist_manager.apply_watchlist_update")
    @patch("src.services.watchlist.watchlist_manager.diff_watchlist")
    @patch("src.services.watchlist.watchlist_manager.fetch_index_symbols")
    @patch("src.services.watchlist.watchlist_manager._load_watchlist")
    def test_runs_for_specified_markets(
        self, mock_load, mock_fetch, mock_diff, mock_apply, mock_save_snapshot
    ):
        """markets 引数で対象マーケットを絞れること"""
        from src.services.watchlist.watchlist_manager import WatchlistDiff, run_watchlist_refresh

        mock_load.return_value = {"jp": ["7203"], "us": ["AAPL"]}
        mock_fetch.return_value = ["7203"]
        mock_diff.return_value = WatchlistDiff(
            market="jp", added=[], kept=["7203"], removed=[], removed_unverified=[], capped=False
        )
        result = run_watchlist_refresh(markets=["jp"])
        assert len(result) <= 1
        mock_save_snapshot.assert_called_once()
