"""ユニットテスト: discord_query_service"""

import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.services.discord_query_service import (
    get_latest_market_prediction_snapshots,
    get_scheduler_job_statuses,
    get_signal_snapshot,
)


class TestDiscordQueryService(unittest.TestCase):
    @patch("src.services.discord_query_service.load_prediction_results")
    @patch("src.services.discord_query_service.load_prediction_markets", return_value=["JP"])
    @patch(
        "src.services.discord_query_service.load_latest_prediction_timestamp",
        return_value="2026-04-06T00:00:00+00:00",
    )
    def test_get_latest_market_prediction_snapshots_builds_market_snapshots(
        self,
        _mock_ts,
        _mock_markets,
        mock_load_results,
    ):
        mock_load_results.side_effect = [
            pd.DataFrame(
                [
                    {
                        "market": "JP",
                        "symbol": "7203",
                        "current_price": 100.0,
                        "avg_pred_price": 101.0,
                        "diff_ratio": 0.01,
                        "model_count": 2,
                    }
                ]
            ),
            pd.DataFrame(),
        ]

        latest_ts, snapshots = get_latest_market_prediction_snapshots()

        self.assertEqual(latest_ts, "2026-04-06T00:00:00+00:00")
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].market, "JP")
        self.assertEqual(snapshots[0].top_results[0].symbol, "7203")
        self.assertEqual(snapshots[0].worst_results, [])

    @patch("src.models.predict_single_stock.explain_prediction_shap")
    @patch("src.models.predict_single_stock.predict_single_stock")
    def test_get_signal_snapshot_maps_prediction_and_shap(
        self,
        mock_predict_single_stock,
        mock_explain_prediction_shap,
    ):
        mock_predict_single_stock.return_value = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "current_price": 100.0,
                    "avg_pred_price": 101.0,
                    "diff_ratio": 0.01,
                    "model_count": 2,
                }
            ]
        )
        mock_explain_prediction_shap.return_value = {
            "direction": "up",
            "top_features": [{"feature": "close_lag_1", "shap_value": 0.5}],
        }

        snapshot = get_signal_snapshot("jp", "7203", explain=True)

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.prediction.symbol, "7203")
        self.assertEqual(snapshot.shap_direction, "up")
        self.assertEqual(snapshot.top_features[0].feature, "close_lag_1")

    def test_get_scheduler_job_statuses_reads_state_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = f"{temp_dir}/scheduler_queue_state.json"
            with open(state_path, "w", encoding="utf-8") as file_handle:
                json.dump(
                    {
                        "events": [
                            {
                                "job_id": "daily_pipeline",
                                "finished_at": "2026-04-06T00:30:00+00:00",
                                "status": "success",
                            }
                        ]
                    },
                    file_handle,
                )

            statuses = get_scheduler_job_statuses(state_path)

        self.assertEqual(statuses[0].job_id, "daily_pipeline")
        self.assertEqual(statuses[0].status, "success")

    # ------------------------------------------------------------------
    # get_monthly_report_summary
    # ------------------------------------------------------------------

    @patch("src.services.monthly_report_pipeline.run_monthly_report")
    def test_get_monthly_report_summary_delegates_to_pipeline(self, mock_run):
        from src.domain.types import MonthlyReportSummary
        from src.services.discord_query_service import get_monthly_report_summary

        expected = MonthlyReportSummary(
            generated_at="2026-04-12T00:00:00",
            target_month="2026-04",
            net_return=0.03,
            max_drawdown=-0.10,
            sharpe_ratio=1.1,
            hit_rate=0.6,
            avg_slippage=0.001,
            symbol_count=10,
            wf_snapshot_file="wf_summary_20260401.csv",
        )
        mock_run.return_value = expected

        result = get_monthly_report_summary("2026-04")

        mock_run.assert_called_once_with(target_month="2026-04")
        self.assertEqual(result.target_month, "2026-04")
        self.assertAlmostEqual(result.net_return, 0.03)

    @patch("src.services.monthly_report_pipeline.run_monthly_report")
    def test_get_monthly_report_summary_passes_none_when_month_omitted(self, mock_run):
        from src.domain.types import MonthlyReportSummary
        from src.services.discord_query_service import get_monthly_report_summary

        mock_run.return_value = MonthlyReportSummary(
            generated_at="2026-04-12T00:00:00",
            target_month="2026-04",
            net_return=None,
            max_drawdown=None,
            sharpe_ratio=None,
            hit_rate=None,
            avg_slippage=None,
        )

        get_monthly_report_summary()

        mock_run.assert_called_once_with(target_month=None)


if __name__ == "__main__":
    unittest.main()
