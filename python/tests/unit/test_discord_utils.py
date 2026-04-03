"""ユニットテスト: discord_utils"""

import unittest
from unittest.mock import patch

from src.api.discord_utils import send_daily_order_completion


class TestSendDailyOrderCompletion(unittest.TestCase):
    @patch("src.api.discord_utils.send_webhook_notification", return_value=True)
    def test_normal_completion_notification(self, mock_send):
        result = send_daily_order_completion(buy_orders=2, sell_orders=1, mode="paper")

        self.assertTrue(result)
        mock_send.assert_called_once()
        title, message = mock_send.call_args.args[:2]
        color = mock_send.call_args.kwargs["color"]

        self.assertEqual(title, "✅ 自動発注完了")
        self.assertIn("モード: paper", message)
        self.assertIn("買い注文: 2 件", message)
        self.assertIn("売り注文: 1 件", message)
        self.assertNotIn("停止理由:", message)
        self.assertEqual(color, 0x00BFFF)

    @patch("src.api.discord_utils.send_webhook_notification", return_value=True)
    def test_stopped_completion_notification_contains_stop_details(self, mock_send):
        result = send_daily_order_completion(
            buy_orders=0,
            sell_orders=0,
            mode="paper",
            trading_stopped=True,
            stop_reason="日次損失上限に到達",
            daily_loss=25_000.0,
            daily_loss_limit=20_000.0,
        )

        self.assertTrue(result)
        mock_send.assert_called_once()
        title, message = mock_send.call_args.args[:2]
        color = mock_send.call_args.kwargs["color"]

        self.assertEqual(title, "⚠️ 自動発注停止")
        self.assertIn("停止理由: 日次損失上限に到達", message)
        self.assertIn("当日損失: 25000 円 / 上限: 20000 円", message)
        self.assertEqual(color, 0xFF9900)


if __name__ == "__main__":
    unittest.main()
