"""ユニットテスト: discord_utils"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.api.discord_utils import (
    send_daily_order_completion,
    send_shap_notification,
    send_weekly_report,
)


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


class TestSendShapNotification(unittest.TestCase):
    @patch("src.api.discord_utils.send_webhook_text", return_value=True)
    def test_sends_top_and_bottom_sections(self, mock_send):
        shap_df = pd.DataFrame(
            {
                "feature": [f"feat_{i}" for i in range(1, 13)],
                "shap_mean": [float(13 - i) / 10 for i in range(1, 13)],
                "shap_rank": list(range(1, 13)),
            }
        )

        result = send_shap_notification("jp", "7203", "StockXGBoostModel", shap_df)

        self.assertTrue(result)
        self.assertGreaterEqual(mock_send.call_count, 1)
        sent_text = "\n".join(call.args[0] for call in mock_send.call_args_list)
        self.assertIn("SHAP特徴量寄与 [jp/7203] StockXGBoostModel", sent_text)
        self.assertIn("上位（寄与大）", sent_text)
        self.assertIn("下位（寄与小）", sent_text)
        self.assertIn("feat_1", sent_text)
        self.assertIn("feat_12", sent_text)

    @patch("src.api.discord_utils.send_webhook_text", return_value=True)
    def test_returns_false_for_empty_dataframe(self, mock_send):
        result = send_shap_notification(
            "jp",
            "7203",
            "StockXGBoostModel",
            pd.DataFrame(columns=["feature", "shap_mean", "shap_rank"]),
        )

        self.assertFalse(result)
        mock_send.assert_not_called()


class TestSendWeeklyReport(unittest.TestCase):
    @patch("src.api.discord_utils.send_webhook_text", return_value=True)
    def test_appends_paper_real_diff_section(self, mock_send):
        accuracy_df = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "direction_accuracy": 0.55,
                    "mean_abs_error": 0.012,
                    "n_samples": 20,
                }
            ]
        )
        diff_summary = {
            "tracked_count": 4,
            "comparable_count": 2,
            "avg_paper_slippage": 0.001,
            "avg_real_slippage": 0.002,
            "avg_abs_price_diff": 3.5,
            "avg_abs_diff_ratio": 0.0035,
            "max_abs_price_diff": 7.0,
        }

        result = send_weekly_report(accuracy_df=accuracy_df, horizon=1, diff_summary=diff_summary)

        self.assertTrue(result)
        sent_text = "\n".join(call.args[0] for call in mock_send.call_args_list)
        self.assertIn("paper/real 乖離サマリー", sent_text)
        self.assertIn("tracked=4件, comparable=2件", sent_text)
        self.assertIn("平均価格差=3.500", sent_text)


if __name__ == "__main__":
    unittest.main()
