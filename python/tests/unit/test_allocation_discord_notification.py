"""ユニットテスト: send_allocation_rebalance_report"""

import unittest
from unittest.mock import patch


class TestSendAllocationRebalanceReport(unittest.TestCase):
    @patch("src.reporting.discord.notifications_model.send_status_fields", return_value=True)
    def test_sends_status_fields_with_expected_spec(self, mock_send):
        from src.reporting.discord.discord_notification_specs import ALLOCATION_REBALANCE_COMPLETION
        from src.reporting.discord.notifications_model import send_allocation_rebalance_report

        result = send_allocation_rebalance_report(
            action="initial",
            tqqq_price=100.0,
            shy_price=50.0,
            tqqq_qty_before=0.0,
            shy_qty_before=0.0,
            cash_before=100_000.0,
            tqqq_qty_after=800.0,
            shy_qty_after=400.0,
            cash_after=0.0,
        )

        self.assertTrue(result)
        mock_send.assert_called_once()
        spec_arg, fields_arg = mock_send.call_args[0][0], mock_send.call_args[0][1]
        self.assertEqual(spec_arg, ALLOCATION_REBALANCE_COMPLETION)
        field_names = [f["name"] for f in fields_arg]
        self.assertIn("TQQQ", field_names)
        self.assertIn("SHY", field_names)

    @patch("src.reporting.discord.notifications_model.send_status_fields", return_value=True)
    def test_rebalance_action_included_in_fields(self, mock_send):
        from src.reporting.discord.notifications_model import send_allocation_rebalance_report

        send_allocation_rebalance_report(
            action="rebalance",
            tqqq_price=120.0,
            shy_price=50.0,
            tqqq_qty_before=800.0,
            shy_qty_before=400.0,
            cash_before=0.0,
            tqqq_qty_after=773.33,
            shy_qty_after=464.0,
            cash_after=0.0,
        )

        fields_arg = mock_send.call_args[0][1]
        type_field = next(f for f in fields_arg if f["name"] == "📌 種別")
        self.assertEqual(type_field["value"], "リバランス")


if __name__ == "__main__":
    unittest.main()
