from unittest.mock import patch

from src.reporting.discord.notifications_model import send_strategy_promotion_detected


class TestSendStrategyPromotionDetected:
    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_sends_message_with_pr_and_hash(self, mock_send):
        mock_send.return_value = True

        result = send_strategy_promotion_detected(
            pr_number=564, rule_or_feature_id="fb44f0011174", pre_promotion_baseline=1.25
        )

        assert result is True
        mock_send.assert_called_once()
        message = mock_send.call_args[0][0]
        assert "564" in message
        assert "fb44f0011174" in message
        assert "1.25" in message

    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_returns_false_on_send_failure(self, mock_send):
        mock_send.return_value = False

        result = send_strategy_promotion_detected(
            pr_number=1, rule_or_feature_id="hash", pre_promotion_baseline=0.5
        )

        assert result is False
