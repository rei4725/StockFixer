"""Unit tests: 指数銘柄取得の User-Agent 付与と取得失敗の可視化

Wikipedia は既定の User-Agent を 403 で拒否するため、明示的な UA を
付けないと指数銘柄取得が恒久的に空振りする（#717）。
"""

import unittest
from unittest.mock import patch

import pandas as pd


class TestFetchPassesUserAgent(unittest.TestCase):
    """read_html に User-Agent が渡ること"""

    @patch("pandas.read_html")
    def test_sp500_passes_user_agent(self, mock_html):
        """S&P500 取得が storage_options で User-Agent を渡すこと"""
        from src.watchlist.manager import fetch_sp500_symbols

        mock_html.return_value = [pd.DataFrame({"Symbol": ["AAPL"]})]
        fetch_sp500_symbols()

        storage_options = mock_html.call_args.kwargs.get("storage_options")
        self.assertIsNotNone(storage_options, "storage_options が渡されていない")
        self.assertIn("User-Agent", storage_options)
        self.assertTrue(storage_options["User-Agent"].strip())

    @patch("pandas.read_html")
    def test_nikkei225_passes_user_agent(self, mock_html):
        """日経225 取得が storage_options で User-Agent を渡すこと"""
        from src.watchlist.manager import fetch_nikkei225_symbols

        mock_html.return_value = [pd.DataFrame({"コード": ["7203"]})]
        fetch_nikkei225_symbols()

        storage_options = mock_html.call_args.kwargs.get("storage_options")
        self.assertIsNotNone(storage_options, "storage_options が渡されていない")
        self.assertIn("User-Agent", storage_options)


class TestFetchFailureIsVisible(unittest.TestCase):
    """取得失敗が WatchlistDiff に現れること"""

    def test_diff_has_fetch_failed_flag(self):
        """WatchlistDiff が fetch_failed を持つこと"""
        from src.watchlist.manager import WatchlistDiff

        self.assertFalse(WatchlistDiff(market="us").fetch_failed)
        self.assertTrue(WatchlistDiff(market="us", fetch_failed=True).fetch_failed)

    @patch("src.watchlist.manager._load_watchlist")
    @patch("src.watchlist.manager.fetch_index_symbols")
    def test_refresh_reports_fetch_failure(self, mock_fetch, mock_load):
        """指数取得が空なら fetch_failed=True の diff が返ること"""
        from src.watchlist.manager import run_watchlist_refresh

        mock_load.return_value = {"us": ["AAPL"], "jp": []}
        mock_fetch.return_value = []

        diffs = run_watchlist_refresh(markets=["us"])

        self.assertEqual(len(diffs), 1, "取得失敗が呼び出し元に伝わっていない")
        self.assertTrue(diffs[0].fetch_failed)
        self.assertFalse(diffs[0].has_changes, "取得失敗時に変更を出してはならない")


if __name__ == "__main__":
    unittest.main()


class TestFetchFailureIsNotified(unittest.TestCase):
    """取得失敗が Discord 通知に現れること"""

    @patch("src.reporting.discord.discord_utils.send_webhook_text_chunked")
    def test_report_warns_on_fetch_failure(self, mock_send):
        """fetch_failed の diff だけでも通知され、警告文が含まれること"""
        from src.reporting.discord.discord_utils import send_watchlist_update_report
        from src.watchlist.manager import WatchlistDiff

        mock_send.return_value = True
        send_watchlist_update_report([WatchlistDiff(market="us", fetch_failed=True)])

        mock_send.assert_called_once()
        body = mock_send.call_args.args[0]
        self.assertIn("取得失敗", body)
        self.assertIn("US", body)

    @patch("src.reporting.discord.discord_utils.send_webhook_text_chunked")
    def test_report_still_skips_when_no_change(self, mock_send):
        """変更も失敗も無ければ従来どおり送信しないこと"""
        from src.reporting.discord.discord_utils import send_watchlist_update_report
        from src.watchlist.manager import WatchlistDiff

        send_watchlist_update_report([WatchlistDiff(market="us")])
        mock_send.assert_not_called()
