from unittest.mock import patch

from src.reporting.discord.notifications_model import (
    send_promotion_result,
    send_strategy_promotion_detected,
)


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


_ALL_LABELS = {
    "net_return": "Net Return",
    "mdd": "MDD",
    "sharpe": "Sharpe",
    "hit_rate": "Hit Rate",
    "slippage": "Slippage",
    "prediction_spread": "Prediction Spread",
}


class TestSendPromotionResult:
    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_promoted_heading(self, mock_send):
        mock_send.return_value = True

        send_promotion_result(promoted=True, reason="全基準を満たした", criteria={})

        message = mock_send.call_args[0][0]
        assert message.splitlines()[0] == "**[週次] モデル昇格評価: 昇格実行**"

    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_not_promoted_heading(self, mock_send):
        mock_send.return_value = True

        send_promotion_result(promoted=False, reason="MDD が基準を超えた", criteria={})

        message = mock_send.call_args[0][0]
        assert message.splitlines()[0] == "**[週次] モデル昇格評価: 見送り**"

    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_reason_in_body(self, mock_send):
        mock_send.return_value = True

        send_promotion_result(promoted=False, reason="Sharpe が champion に届かない", criteria={})

        lines = mock_send.call_args[0][0].splitlines()
        assert "Sharpe が champion に届かない" in lines

    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_passed_criteria_are_ok_with_labels(self, mock_send):
        mock_send.return_value = True
        criteria = {key: {"passed": True} for key in _ALL_LABELS}

        send_promotion_result(promoted=True, reason="r", criteria=criteria)

        lines = mock_send.call_args[0][0].splitlines()
        for label in _ALL_LABELS.values():
            assert f"[OK] {label}" in lines

    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_failed_criteria_are_ng_with_labels(self, mock_send):
        mock_send.return_value = True
        criteria = {key: {"passed": False} for key in _ALL_LABELS}

        send_promotion_result(promoted=False, reason="r", criteria=criteria)

        lines = mock_send.call_args[0][0].splitlines()
        for label in _ALL_LABELS.values():
            assert f"[NG] {label}" in lines

    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_mixed_criteria_in_order_inside_code_block(self, mock_send):
        mock_send.return_value = True
        criteria = {
            "net_return": {"passed": True},
            "mdd": {"passed": False},
        }

        send_promotion_result(promoted=False, reason="r", criteria=criteria)

        lines = mock_send.call_args[0][0].splitlines()
        assert lines[-4:] == ["```", "[OK] Net Return", "[NG] MDD", "```"]

    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_unknown_key_uses_key_name(self, mock_send):
        mock_send.return_value = True
        criteria = {"custom_metric": {"passed": True}}

        send_promotion_result(promoted=True, reason="r", criteria=criteria)

        lines = mock_send.call_args[0][0].splitlines()
        assert "[OK] custom_metric" in lines

    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_returns_send_result(self, mock_send):
        mock_send.return_value = False
        assert send_promotion_result(promoted=True, reason="r", criteria={}) is False

        mock_send.return_value = True
        assert send_promotion_result(promoted=True, reason="r", criteria={}) is True
        assert mock_send.call_count == 2
