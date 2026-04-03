"""
Unit Test: src.utils.retry_helper.with_retry

外部依存（実ネットワーク・time.sleep）なし、全 Mock で高速実行。
- 正常系: 初回成功・リトライ後成功
- 異常系: 非リトライエラーでの即 raise・max_retries 超過
- バックオフ: 待機時間の指数増加・上限キャップ
"""
from unittest.mock import MagicMock, patch

import pytest

from src.utils.retry_helper import (
    _countdown_wait,
    retry_ticker_history,
    retry_yfinance_download,
    with_retry,
)


class TestWithRetrySuccess:
    """正常系: 成功パターン"""

    def test_succeeds_on_first_attempt(self):
        """初回で成功した場合はリトライなし、戻り値を返す"""
        func = MagicMock(return_value="ok")
        result = with_retry(func, max_retries=3)
        assert result == "ok"
        func.assert_called_once()

    def test_succeeds_after_one_retry_on_rate_limit(self):
        """レート制限エラー1回 → 2回目で成功"""
        func = MagicMock(side_effect=[Exception("rate limit exceeded"), "ok"])
        with patch("src.utils.retry_helper.time.sleep"):
            result = with_retry(func, max_retries=3, base_wait_seconds=0.01)
        assert result == "ok"
        assert func.call_count == 2

    def test_succeeds_after_retry_on_too_many_requests(self):
        """429 Too Many Requests → リトライで成功"""
        func = MagicMock(side_effect=[Exception("429 too many requests"), "done"])
        with patch("src.utils.retry_helper.time.sleep"):
            result = with_retry(func, max_retries=3, base_wait_seconds=0.01)
        assert result == "done"

    def test_succeeds_after_retry_on_connection_error(self):
        """接続エラー → リトライで成功"""
        func = MagicMock(side_effect=[Exception("connection reset"), 42])
        with patch("src.utils.retry_helper.time.sleep"):
            result = with_retry(func, max_retries=3, base_wait_seconds=0.01)
        assert result == 42

    def test_succeeds_after_retry_on_timeout(self):
        """タイムアウト → リトライで成功"""
        func = MagicMock(side_effect=[Exception("timeout occurred"), "result"])
        with patch("src.utils.retry_helper.time.sleep"):
            result = with_retry(func, max_retries=3, base_wait_seconds=0.01)
        assert result == "result"


class TestWithRetryFailure:
    """異常系: エラーで失敗するパターン"""

    def test_raises_immediately_on_non_retriable_error(self):
        """リトライ対象外エラー（ValueError 等）は即 raise・1回だけ呼ばれる"""
        func = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            with_retry(func, max_retries=5)
        func.assert_called_once()

    def test_raises_immediately_on_key_error(self):
        """Non-retriable error (KeyError) → raises immediately without retry."""
        func = MagicMock(side_effect=KeyError("missing_key"))
        with pytest.raises(KeyError):
            with_retry(func, max_retries=3)
        func.assert_called_once()

    def test_raises_after_max_retries_on_rate_limit(self):
        """レート制限が max_retries 回続いた場合は最後に raise"""
        func = MagicMock(side_effect=Exception("429 too many requests"))
        with patch("src.utils.retry_helper.time.sleep"), pytest.raises(Exception):
            with_retry(func, max_retries=3, base_wait_seconds=0.01)
        assert func.call_count == 3

    def test_raises_after_max_retries_on_network_error(self):
        """ネットワークエラーが max_retries 回続いた場合は最後に raise"""
        func = MagicMock(side_effect=Exception("connection refused"))
        with patch("src.utils.retry_helper.time.sleep"), pytest.raises(Exception):
            with_retry(func, max_retries=2, base_wait_seconds=0.01)
        assert func.call_count == 2

    def test_raises_after_max_retries_on_temporarily_unavailable(self):
        """Temporarily unavailable → raises after max_retries is exhausted."""
        func = MagicMock(side_effect=Exception("temporarily unavailable"))
        with patch("src.utils.retry_helper.time.sleep"), pytest.raises(Exception):
            with_retry(func, max_retries=2, base_wait_seconds=0.01)
        assert func.call_count == 2


class TestWithRetryBackoff:
    """バックオフ: 待機時間の検証（base_wait_seconds < 5.0 で time.sleep を使用）"""

    def test_exponential_wait_time_grows(self):
        """待機時間が指数的に増加する（base * 2^(n-1)）"""
        func = MagicMock(
            side_effect=[
                Exception("rate limit"),
                Exception("rate limit"),
                "ok",
            ]
        )
        sleep_calls = []
        with patch(
            "src.utils.retry_helper.time.sleep", side_effect=lambda t: sleep_calls.append(t)
        ):
            with_retry(func, max_retries=5, base_wait_seconds=0.5)

        assert len(sleep_calls) == 2
        # 1回目: 0.5秒、2回目: 1.0秒（指数増加）
        assert sleep_calls[0] == pytest.approx(0.5)
        assert sleep_calls[1] == pytest.approx(1.0)

    def test_wait_time_capped_at_max(self):
        """待機時間は max_wait_seconds を超えない"""
        func = MagicMock(
            side_effect=[
                Exception("rate limit"),
                Exception("rate limit"),
                Exception("rate limit"),
                "ok",
            ]
        )
        sleep_calls = []
        with patch(
            "src.utils.retry_helper.time.sleep", side_effect=lambda t: sleep_calls.append(t)
        ):
            with_retry(func, max_retries=5, base_wait_seconds=1.0, max_wait_seconds=2.0)

        # 1回目: 1.0秒, 2回目: 2.0秒, 3回目: min(4.0, 2.0) = 2.0秒（キャップ）
        assert all(t <= 2.0 for t in sleep_calls)
        assert sleep_calls[-1] == pytest.approx(2.0)

    def test_sleep_called_before_each_retry(self):
        """リトライのたびに sleep が呼ばれる（呼び出し回数 = リトライ回数）"""
        func = MagicMock(
            side_effect=[
                Exception("rate limit"),
                Exception("rate limit"),
                "ok",
            ]
        )
        with patch("src.utils.retry_helper.time.sleep") as mock_sleep:
            with_retry(func, max_retries=5, base_wait_seconds=0.1)
        assert mock_sleep.call_count == 2


class TestCountdownWait:
    """_countdown_wait のテスト（5秒以上の待機時のカウントダウン表示付き待機）"""

    def test_sleeps_for_each_second(self):
        """整数秒ごとに time.sleep(1) が呼ばれること"""
        with patch("src.utils.retry_helper.time.sleep") as mock_sleep:
            _countdown_wait(3.0)
        assert mock_sleep.call_count == 3

    def test_zero_seconds_no_sleep(self):
        """0秒では sleep が呼ばれないこと"""
        with patch("src.utils.retry_helper.time.sleep") as mock_sleep:
            _countdown_wait(0.0)
        mock_sleep.assert_not_called()

    def test_fractional_seconds_truncated(self):
        """小数秒は切り捨てて整数秒ぶん sleep すること"""
        with patch("src.utils.retry_helper.time.sleep") as mock_sleep:
            _countdown_wait(2.9)
        assert mock_sleep.call_count == 2

    def test_countdown_triggered_for_5s_wait(self):
        """wait_time >= 5.0 のとき _countdown_wait が使われること（統合確認）"""
        func = MagicMock(side_effect=[Exception("rate limit exceeded"), "ok"])
        with patch("src.utils.retry_helper._countdown_wait") as mock_countdown:
            with_retry(func, max_retries=3, base_wait_seconds=5.0, max_wait_seconds=60.0)
        mock_countdown.assert_called_once()


class TestRetryYfinanceDownload:
    """retry_yfinance_download のテスト"""

    def test_calls_yf_download(self):
        """yf.download を呼び出すこと"""
        import pandas as pd

        mock_df = pd.DataFrame({"Close": [100.0]})
        with patch("src.utils.retry_helper.yf.download", return_value=mock_df) as mock_dl:
            result = retry_yfinance_download("AAPL", "2026-01-01", "2026-01-31")
        mock_dl.assert_called_once()
        assert result is mock_df

    def test_retries_on_rate_limit(self):
        """レート制限エラーでリトライしてから成功すること"""
        import pandas as pd

        mock_df = pd.DataFrame({"Close": [100.0]})
        with patch(
            "src.utils.retry_helper.yf.download",
            side_effect=[Exception("rate limit exceeded"), mock_df],
        ), patch("src.utils.retry_helper.time.sleep"):
            result = retry_yfinance_download("AAPL", "2026-01-01", "2026-01-31")
        assert result is mock_df


class TestRetryTickerHistory:
    """retry_ticker_history のテスト"""

    def test_calls_ticker_history(self):
        """ticker_obj.history を呼び出すこと"""
        import pandas as pd

        mock_df = pd.DataFrame({"Close": [100.0]})
        ticker_obj = MagicMock()
        ticker_obj.history.return_value = mock_df

        result = retry_ticker_history(ticker_obj, "2026-01-01", "2026-01-31")
        ticker_obj.history.assert_called_once()
        assert result is mock_df

    def test_retries_on_connection_error(self):
        """接続エラーでリトライしてから成功すること"""
        import pandas as pd

        mock_df = pd.DataFrame({"Close": [100.0]})
        ticker_obj = MagicMock()
        ticker_obj.history.side_effect = [Exception("connection reset"), mock_df]

        with patch("src.utils.retry_helper.time.sleep"):
            result = retry_ticker_history(ticker_obj, "2026-01-01", "2026-01-31")
        assert result is mock_df
