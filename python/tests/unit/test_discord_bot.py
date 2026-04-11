"""ユニットテスト: discord_bot の純粋関数"""

import unittest

from src.api.discord_bot import (
    TOP10_LABEL,
    build_prediction_table_text,
    build_ranked_stock_message,
    build_scheduler_status_lines,
    build_signal_lines,
    determine_signal_label,
)
from src.domain.types import PredictionResult, SchedulerJobStatus


class TestDiscordBotHelpers(unittest.TestCase):
    def test_build_ranked_stock_message_wraps_table_in_code_block(self):
        message = build_ranked_stock_message("JP", TOP10_LABEL, "7203  +1.2%")

        self.assertIn("=== JP 差異割合上位10銘柄 ===", message)
        self.assertIn("```text", message)
        self.assertIn("7203  +1.2%", message)

    def test_determine_signal_label_uses_thresholds(self):
        self.assertEqual(determine_signal_label(0.01), "⬆️ Buy")
        self.assertEqual(determine_signal_label(-0.01), "⬇️ Sell")
        self.assertEqual(determine_signal_label(0.001), "⏺️ Hold")

    def test_build_signal_lines_includes_shap_section(self):
        lines = build_signal_lines(
            market="jp",
            symbol="7203",
            current_price=100.1234,
            predicted_price=101.5678,
            diff_ratio=0.01,
            model_count=3,
            shap_result={
                "direction": "up",
                "top_features": [
                    {"feature": "close_lag_1", "shap_value": 0.12345},
                ],
            },
        )

        message = "\n".join(lines)
        self.assertIn("=== JP / 7203 ===", message)
        self.assertIn("シグナル : ⬆️ Buy", message)
        self.assertIn("[SHAP 寄与度 Top5 - 方向:up]", message)
        self.assertIn("close_lag_1", message)

    def test_build_prediction_table_text_formats_prediction_results(self):
        table_text = build_prediction_table_text(
            [
                PredictionResult(
                    market="jp",
                    symbol="7203",
                    current_price=100.0,
                    avg_pred_price=101.5,
                    diff_ratio=0.015,
                    model_count=2,
                )
            ]
        )

        self.assertIn("シンボル", table_text)
        self.assertIn("7203", table_text)
        self.assertIn("+1.5%", table_text)

    def test_build_scheduler_status_lines_formats_latest_runs(self):
        lines = build_scheduler_status_lines(
            [
                SchedulerJobStatus(
                    job_id="daily_pipeline",
                    label="日次 (daily_pipeline)",
                    last_run_at="2026-04-06T00:30:00+00:00",
                    status="success",
                )
            ]
        )

        message = "\n".join(lines)
        self.assertIn("=== スケジューラ状態 ===", message)
        self.assertIn("日次 (daily_pipeline)", message)
        self.assertIn("2026-04-06 09:30:00 JST", message)
        self.assertIn("[状態: success]", message)


if __name__ == "__main__":
    unittest.main()
