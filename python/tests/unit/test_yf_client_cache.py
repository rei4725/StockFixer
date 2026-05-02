"""yf_client キャッシュ層のユニットテスト（#149）"""

import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


def _make_df(n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": [100.0 + i for i in range(n)]}, index=dates)


class TestCacheTtl(unittest.TestCase):
    """_cache_ttl の TTL 判定ロジックのテスト"""

    def test_past_end_date_returns_long_ttl(self):
        from src.market_data.yf_client import _TTL_LONG, _cache_ttl

        self.assertEqual(_cache_ttl("2020-01-01"), _TTL_LONG)

    def test_today_end_date_returns_short_ttl(self):
        from src.market_data.yf_client import _TTL_SHORT, _cache_ttl

        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        self.assertEqual(_cache_ttl(today), _TTL_SHORT)

    def test_future_end_date_returns_short_ttl(self):
        from src.market_data.yf_client import _TTL_SHORT, _cache_ttl

        future = (pd.Timestamp.now() + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        self.assertEqual(_cache_ttl(future), _TTL_SHORT)

    def test_none_end_date_returns_short_ttl(self):
        from src.market_data.yf_client import _TTL_SHORT, _cache_ttl

        self.assertEqual(_cache_ttl(None), _TTL_SHORT)


class _CacheTestBase(unittest.TestCase):
    """キャッシュ系テストの共通セットアップ"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self._patch_data_dir = patch(
            "src.market_data.yf_client.get_data_dir", return_value=self.tmp_dir
        )
        self._patch_data_dir.start()
        import src.market_data.yf_client as mod

        mod.clear_cache()
        mod.set_no_cache(False)

    def tearDown(self):
        self._patch_data_dir.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        import src.market_data.yf_client as mod

        mod.clear_cache()
        mod.set_no_cache(False)


class TestDownloadCache(_CacheTestBase):
    """download() のキャッシュ動作テスト"""

    @patch("src.market_data.yf_client.with_retry")
    def test_second_call_uses_memory_cache(self, mock_retry):
        """同じ引数での2回目呼び出しはキャッシュから返すこと"""
        mock_retry.return_value = _make_df()

        import src.market_data.yf_client as yf_client

        r1 = yf_client.download("AAPL", start="2024-01-01", end="2024-06-01")
        r2 = yf_client.download("AAPL", start="2024-01-01", end="2024-06-01")

        self.assertEqual(mock_retry.call_count, 1)
        pd.testing.assert_frame_equal(r1, r2)

    @patch("src.market_data.yf_client.with_retry")
    def test_different_params_fetch_separately(self, mock_retry):
        """異なる引数は別々に fetch されること"""
        mock_retry.return_value = _make_df()

        import src.market_data.yf_client as yf_client

        yf_client.download("AAPL", start="2024-01-01", end="2024-06-01")
        yf_client.download("AAPL", start="2024-01-01", end="2024-09-01")

        self.assertEqual(mock_retry.call_count, 2)

    @patch("src.market_data.yf_client.with_retry")
    def test_no_cache_flag_bypasses_cache(self, mock_retry):
        """set_no_cache(True) のとき毎回 fetch されること"""
        mock_retry.return_value = _make_df()

        import src.market_data.yf_client as yf_client

        yf_client.set_no_cache(True)
        yf_client.download("AAPL", start="2024-01-01", end="2024-06-01")
        yf_client.download("AAPL", start="2024-01-01", end="2024-06-01")

        self.assertEqual(mock_retry.call_count, 2)

    @patch("src.market_data.yf_client.with_retry")
    def test_empty_df_not_cached(self, mock_retry):
        """空 DataFrame はキャッシュしないこと"""
        mock_retry.return_value = pd.DataFrame()

        import src.market_data.yf_client as yf_client

        yf_client.download("INVALID", start="2024-01-01", end="2024-06-01")
        yf_client.download("INVALID", start="2024-01-01", end="2024-06-01")

        self.assertEqual(mock_retry.call_count, 2)

    @patch("src.market_data.yf_client.with_retry")
    def test_period_request_not_cached(self, mock_retry):
        """period 指定（相対時間）はキャッシュしないこと"""
        mock_retry.return_value = _make_df()

        import src.market_data.yf_client as yf_client

        yf_client.download("AAPL", period="1mo")
        yf_client.download("AAPL", period="1mo")

        self.assertEqual(mock_retry.call_count, 2)

    @patch("src.market_data.yf_client.with_retry")
    def test_clear_cache_invalidates_memory_cache(self, mock_retry):
        """clear_cache() でキャッシュクリア後は再 fetch されること"""
        mock_retry.return_value = _make_df()

        import src.market_data.yf_client as yf_client

        yf_client.download("AAPL", start="2024-01-01", end="2024-06-01")
        yf_client.clear_cache()
        yf_client.download("AAPL", start="2024-01-01", end="2024-06-01")

        self.assertEqual(mock_retry.call_count, 2)

    @patch("src.market_data.yf_client.with_retry")
    def test_disk_cache_used_after_memory_eviction(self, mock_retry):
        """メモリキャッシュ削除後もディスクキャッシュから復元されること"""
        mock_retry.return_value = _make_df()

        import src.market_data.yf_client as mod

        mod.download("AAPL", start="2024-01-01", end="2024-06-01")
        # メモリキャッシュのみクリア（ディスクキャッシュは残す）
        with mod._cache_lock:
            mod._memory_cache.clear()

        r2 = mod.download("AAPL", start="2024-01-01", end="2024-06-01")

        # ディスクキャッシュから復元されるので with_retry は1回だけ
        self.assertEqual(mock_retry.call_count, 1)
        self.assertFalse(r2.empty)


class TestTickerHistoryCache(_CacheTestBase):
    """ticker_history() のキャッシュ動作テスト"""

    @patch("src.market_data.yf_client.with_retry")
    def test_second_call_uses_cache(self, mock_retry):
        """2回目呼び出しはキャッシュから返すこと"""
        mock_retry.return_value = _make_df()

        import src.market_data.yf_client as yf_client

        yf_client.ticker_history("AAPL", start="2024-01-01", end="2024-06-01")
        yf_client.ticker_history("AAPL", start="2024-01-01", end="2024-06-01")

        self.assertEqual(mock_retry.call_count, 1)

    @patch("src.market_data.yf_client.with_retry")
    def test_no_cache_flag_bypasses_cache(self, mock_retry):
        """set_no_cache(True) のとき毎回 fetch されること"""
        mock_retry.return_value = _make_df()

        import src.market_data.yf_client as yf_client

        yf_client.set_no_cache(True)
        yf_client.ticker_history("AAPL", start="2024-01-01", end="2024-06-01")
        yf_client.ticker_history("AAPL", start="2024-01-01", end="2024-06-01")

        self.assertEqual(mock_retry.call_count, 2)

    @patch("src.market_data.yf_client.with_retry")
    def test_empty_df_not_cached(self, mock_retry):
        """空 DataFrame はキャッシュしないこと"""
        mock_retry.return_value = pd.DataFrame()

        import src.market_data.yf_client as yf_client

        yf_client.ticker_history("INVALID", start="2024-01-01", end="2024-06-01")
        yf_client.ticker_history("INVALID", start="2024-01-01", end="2024-06-01")

        self.assertEqual(mock_retry.call_count, 2)


if __name__ == "__main__":
    unittest.main()
