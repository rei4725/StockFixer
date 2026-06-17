"""ユニットテスト: Discord レート制限・デデュープ"""

from unittest.mock import MagicMock, patch

from src.reporting.discord.rate_limiter import DiscordRateLimiter


class TestDedupeCache:
    def test_first_send_is_allowed(self):
        limiter = DiscordRateLimiter(ttl=60.0)
        should_send, summary = limiter.check_and_record("エラーメッセージ")
        assert should_send is True
        assert summary is None

    def test_second_send_within_ttl_is_suppressed(self):
        limiter = DiscordRateLimiter(ttl=60.0)
        limiter.check_and_record("エラーメッセージ")
        should_send, summary = limiter.check_and_record("エラーメッセージ")
        assert should_send is False
        assert summary is None

    def test_suppression_counter_increments(self):
        limiter = DiscordRateLimiter(ttl=60.0)
        limiter.check_and_record("エラー")
        limiter.check_and_record("エラー")
        limiter.check_and_record("エラー")
        key = limiter._hash_message("エラー")
        assert limiter._cache[key].suppressed_count == 2

    def test_different_messages_are_not_deduped(self):
        limiter = DiscordRateLimiter(ttl=60.0)
        limiter.check_and_record("エラーA")
        should_send, _ = limiter.check_and_record("エラーB")
        assert should_send is True

    def test_send_allowed_after_ttl_expires(self):
        limiter = DiscordRateLimiter(ttl=0.01)
        limiter.check_and_record("エラー")
        import time

        time.sleep(0.02)
        should_send, _ = limiter.check_and_record("エラー")
        assert should_send is True

    def test_suppression_summary_returned_after_ttl(self):
        limiter = DiscordRateLimiter(ttl=0.01)
        limiter.check_and_record("エラー")
        limiter.check_and_record("エラー")  # suppressed_count=1
        limiter.check_and_record("エラー")  # suppressed_count=2
        import time

        time.sleep(0.02)
        should_send, summary = limiter.check_and_record("エラー")
        assert should_send is True
        assert summary is not None
        assert "2" in summary

    def test_no_summary_when_no_suppressions_before_ttl(self):
        limiter = DiscordRateLimiter(ttl=0.01)
        limiter.check_and_record("エラー")
        import time

        time.sleep(0.02)
        should_send, summary = limiter.check_and_record("エラー")
        assert should_send is True
        assert summary is None

    def test_cache_reset_after_ttl(self):
        limiter = DiscordRateLimiter(ttl=0.01)
        limiter.check_and_record("エラー")
        import time

        time.sleep(0.02)
        limiter.check_and_record("エラー")
        key = limiter._hash_message("エラー")
        assert limiter._cache[key].suppressed_count == 0


class TestRateLimit:
    def test_apply_rate_limit_sleeps_when_too_fast(self):
        limiter = DiscordRateLimiter(rate_interval=0.1)
        import time

        limiter.apply_rate_limit()
        start = time.monotonic()
        limiter.apply_rate_limit()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.05  # 少なくとも半分は待つ

    def test_no_sleep_when_interval_elapsed(self):
        limiter = DiscordRateLimiter(rate_interval=0.01)
        import time

        limiter.apply_rate_limit()
        time.sleep(0.02)
        start = time.monotonic()
        limiter.apply_rate_limit()
        elapsed = time.monotonic() - start
        assert elapsed < 0.02  # ほぼ待たない


class TestIntegrationWithSendWebhookNotification:
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    @patch(
        "src.reporting.discord.webhook_sender.isoformat_jst",
        return_value="2026-01-01T00:00:00+09:00",
    )
    @patch("src.reporting.discord.rate_limiter._limiter")
    def test_suppressed_notification_returns_true(self, mock_limiter, mock_ts, mock_post):
        from src.reporting.discord.discord_utils import send_webhook_notification

        mock_limiter.check_and_record.return_value = (False, None)
        result = send_webhook_notification("タイトル", "メッセージ")
        assert result is True
        mock_post.assert_not_called()

    @patch("src.reporting.discord.webhook_sender._post_webhook")
    @patch(
        "src.reporting.discord.webhook_sender.isoformat_jst",
        return_value="2026-01-01T00:00:00+09:00",
    )
    @patch("src.reporting.discord.rate_limiter._limiter")
    def test_suppression_summary_is_sent_before_main(self, mock_limiter, mock_ts, mock_post):
        from src.reporting.discord.discord_utils import send_webhook_notification

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        mock_limiter.check_and_record.return_value = (
            True,
            "[通知抑止サマリー] 同一通知を 3 件抑止しました",
        )

        result = send_webhook_notification("タイトル", "メッセージ")

        assert result is True
        assert mock_post.call_count == 2
        first_call = mock_post.call_args_list[0]
        assert "通知抑止サマリー" in first_call.kwargs["json_payload"]["content"]

    @patch("src.reporting.discord.webhook_sender._post_webhook")
    @patch(
        "src.reporting.discord.webhook_sender.isoformat_jst",
        return_value="2026-01-01T00:00:00+09:00",
    )
    @patch("src.reporting.discord.rate_limiter._limiter")
    def test_normal_send_without_summary(self, mock_limiter, mock_ts, mock_post):
        from src.reporting.discord.discord_utils import send_webhook_notification

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        mock_limiter.check_and_record.return_value = (True, None)

        result = send_webhook_notification("タイトル", "メッセージ")

        assert result is True
        assert mock_post.call_count == 1
