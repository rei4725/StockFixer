"""ユニットテスト: discord_query_service"""

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.reporting.query_service import (
    get_latest_market_prediction_snapshots,
    get_ranked_prediction_results,
    get_scheduler_job_statuses,
    get_signal_snapshot,
    get_watchlist_prediction_view,
)


class TestDiscordQueryService(unittest.TestCase):
    @patch("src.reporting.query_service.load_prediction_results")
    @patch("src.reporting.query_service.load_prediction_markets", return_value=["JP"])
    @patch(
        "src.reporting.query_service.load_latest_prediction_timestamp",
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

    @patch("src.prediction.predict_single.explain_prediction_shap")
    @patch("src.prediction.predict_single.predict_single_stock")
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

    @patch("src.reporting.monthly.run_monthly_report")
    def test_get_monthly_report_summary_delegates_to_pipeline(self, mock_run):
        from src.reporting.query_service import get_monthly_report_summary
        from src.reporting.types import MonthlyReportSummary

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

    @patch("src.reporting.monthly.run_monthly_report")
    def test_get_monthly_report_summary_passes_none_when_month_omitted(self, mock_run):
        from src.reporting.query_service import get_monthly_report_summary
        from src.reporting.types import MonthlyReportSummary

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

    # ------------------------------------------------------------------
    # get_latest_market_prediction_snapshots — no timestamp
    # ------------------------------------------------------------------

    @patch(
        "src.reporting.query_service.load_latest_prediction_timestamp",
        return_value=None,
    )
    def test_get_latest_market_prediction_snapshots_returns_none_when_no_timestamp(self, _mock):
        ts, snapshots = get_latest_market_prediction_snapshots()
        self.assertIsNone(ts)
        self.assertEqual(snapshots, [])

    # ------------------------------------------------------------------
    # get_ranked_prediction_results — else branch (all)
    # ------------------------------------------------------------------

    @patch("src.reporting.query_service.load_prediction_results", return_value=pd.DataFrame())
    def test_get_ranked_prediction_results_all_mode(self, mock_load):
        result = get_ranked_prediction_results("us", "other")
        self.assertEqual(result, [])
        mock_load.assert_called_once()

    # ------------------------------------------------------------------
    # get_signal_snapshot — edge cases
    # ------------------------------------------------------------------

    @patch("src.prediction.predict_single.predict_single_stock", return_value=None)
    def test_get_signal_snapshot_returns_none_when_no_prediction(self, _mock):
        result = get_signal_snapshot("us", "AAPL")
        self.assertIsNone(result)

    @patch("src.prediction.predict_single.predict_single_stock")
    def test_get_signal_snapshot_without_explain(self, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            [
                {
                    "market": "us",
                    "symbol": "AAPL",
                    "current_price": 150.0,
                    "avg_pred_price": 152.0,
                    "diff_ratio": 0.013,
                    "model_count": 2,
                }
            ]
        )
        result = get_signal_snapshot("us", "AAPL", explain=False)
        self.assertIsNotNone(result)
        self.assertIsNone(result.shap_direction)

    @patch("src.prediction.predict_single.explain_prediction_shap", return_value=None)
    @patch("src.prediction.predict_single.predict_single_stock")
    def test_get_signal_snapshot_without_shap_result(self, mock_predict, _mock_shap):
        mock_predict.return_value = pd.DataFrame(
            [
                {
                    "market": "us",
                    "symbol": "AAPL",
                    "current_price": 150.0,
                    "avg_pred_price": 152.0,
                    "diff_ratio": 0.013,
                    "model_count": 2,
                }
            ]
        )
        result = get_signal_snapshot("us", "AAPL", explain=True)
        self.assertIsNotNone(result)
        self.assertIsNone(result.shap_direction)

    # ------------------------------------------------------------------
    # get_watchlist_prediction_view
    # ------------------------------------------------------------------

    def _write_watchlist_csv(self, tmp_dir, rows):
        path = os.path.join(tmp_dir, "monitor_list.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for row in rows:
                writer.writerow(row)
        return path

    @patch("src.prediction.predict_single.predict_single_stock")
    def test_get_watchlist_prediction_view_normal(self, mock_predict):
        mock_predict.return_value = pd.DataFrame(
            [
                {
                    "symbol": "AAPL",
                    "current_price": 150.0,
                    "avg_pred_price": 152.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = self._write_watchlist_csv(tmp_dir, [["us", "AAPL"]])
            with patch("src.reporting.query_service.get_monitor_list_path", return_value=csv_path):
                view = get_watchlist_prediction_view()
        self.assertIsNone(view.error_message)
        self.assertEqual(len(view.rows), 1)
        self.assertEqual(view.rows[0].symbol, "AAPL")

    @patch("src.prediction.predict_single.predict_single_stock", return_value=None)
    def test_get_watchlist_prediction_view_no_prediction(self, _mock):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = self._write_watchlist_csv(tmp_dir, [["us", "AAPL"]])
            with patch("src.reporting.query_service.get_monitor_list_path", return_value=csv_path):
                view = get_watchlist_prediction_view()
        self.assertIsNone(view.error_message)
        self.assertEqual(len(view.rows), 1)
        self.assertIsNone(view.rows[0].current_price)

    def test_get_watchlist_prediction_view_skips_short_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = self._write_watchlist_csv(tmp_dir, [["us"], ["us", "AAPL"]])
            with patch("src.reporting.query_service.get_monitor_list_path", return_value=csv_path):
                with patch("src.prediction.predict_single.predict_single_stock", return_value=None):
                    view = get_watchlist_prediction_view()
        self.assertEqual(len(view.rows), 1)

    def test_get_watchlist_prediction_view_error_returns_error_message(self):
        with patch(
            "src.reporting.query_service.get_monitor_list_path",
            return_value="/nonexistent/path.csv",
        ):
            view = get_watchlist_prediction_view()
        self.assertIsNotNone(view.error_message)
        self.assertIn("エラー", view.error_message)


if __name__ == "__main__":
    unittest.main()
