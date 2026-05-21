"""ユニットテスト: discord_bot の純粋関数"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.orchestration.types import SchedulerJobStatus
from src.prediction.types import PredictionResult
from src.reporting.discord.discord_bot import (
    TOP10_LABEL,
    build_monthly_report_lines,
    build_prediction_table_text,
    build_ranked_stock_message,
    build_scheduler_status_lines,
    build_signal_lines,
    determine_signal_label,
)
from src.reporting.types import MonthlyReportSummary


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

    # ------------------------------------------------------------------
    # build_monthly_report_lines
    # ------------------------------------------------------------------

    def _make_summary(self, **kwargs):
        defaults = dict(
            generated_at="2026-04-12T10:00:00",
            target_month="2026-04",
            net_return=0.04,
            max_drawdown=-0.10,
            sharpe_ratio=1.2,
            hit_rate=0.65,
            avg_slippage=0.001,
            symbol_count=15,
            wf_snapshot_file="wf_summary_20260401.csv",
        )
        defaults.update(kwargs)
        return MonthlyReportSummary(**defaults)

    def test_build_monthly_report_lines_contains_header(self):
        lines = build_monthly_report_lines(self._make_summary())
        text = "\n".join(lines)
        self.assertIn("=== 月次KPIレポート 2026-04 ===", text)

    def test_build_monthly_report_lines_formats_kpis_as_percent(self):
        lines = build_monthly_report_lines(self._make_summary(net_return=0.04, max_drawdown=-0.10))
        text = "\n".join(lines)
        self.assertIn("4.00%", text)
        self.assertIn("-10.00%", text)

    def test_build_monthly_report_lines_shows_sharpe_achievement(self):
        # sharpe >= 1.0 → 達成
        lines_ok = build_monthly_report_lines(self._make_summary(sharpe_ratio=1.2))
        self.assertIn("達成", "\n".join(lines_ok))

        # sharpe < 1.0 → 未達
        lines_ng = build_monthly_report_lines(self._make_summary(sharpe_ratio=0.8))
        self.assertIn("未達", "\n".join(lines_ng))

    def test_build_monthly_report_lines_shows_mdd_achievement(self):
        # |mdd| <= 0.15 → 達成
        lines_ok = build_monthly_report_lines(self._make_summary(max_drawdown=-0.10))
        self.assertIn("達成", "\n".join(lines_ok))

        # |mdd| > 0.15 → 未達
        lines_ng = build_monthly_report_lines(self._make_summary(max_drawdown=-0.20))
        self.assertIn("未達", "\n".join(lines_ng))

    def test_build_monthly_report_lines_handles_none_kpis(self):
        lines = build_monthly_report_lines(
            self._make_summary(
                net_return=None,
                max_drawdown=None,
                sharpe_ratio=None,
                hit_rate=None,
                avg_slippage=None,
            )
        )
        text = "\n".join(lines)
        self.assertIn("データなし", text)

    def test_build_monthly_report_lines_shows_meta_info(self):
        lines = build_monthly_report_lines(self._make_summary())
        text = "\n".join(lines)
        self.assertIn("15", text)
        self.assertIn("wf_summary_20260401.csv", text)
        self.assertIn("2026-04-12T10:00:00"[:19], text)

    # ------------------------------------------------------------------
    # handle_monthlyreport_command (async)
    # ------------------------------------------------------------------

    @patch("src.reporting.discord.discord_bot.get_monthly_report_summary")
    def test_handle_monthlyreport_command_sends_code_block(self, mock_get):
        from src.reporting.discord.discord_bot import handle_monthlyreport_command

        mock_get.return_value = self._make_summary()

        message = MagicMock()
        message.content = "/monthlyreport"
        message.channel = MagicMock()
        message.channel.send = AsyncMock()

        asyncio.run(handle_monthlyreport_command(message))

        mock_get.assert_called_once_with(None)
        calls = message.channel.send.call_args_list
        full_text = " ".join(str(c) for c in calls)
        self.assertIn("月次KPIレポート", full_text)

    @patch("src.reporting.discord.discord_bot.get_monthly_report_summary")
    def test_handle_monthlyreport_command_passes_month_arg(self, mock_get):
        from src.reporting.discord.discord_bot import handle_monthlyreport_command

        mock_get.return_value = self._make_summary(target_month="2026-03")

        message = MagicMock()
        message.content = "/monthlyreport 2026-03"
        message.channel = MagicMock()
        message.channel.send = AsyncMock()

        asyncio.run(handle_monthlyreport_command(message))

        mock_get.assert_called_once_with("2026-03")

    @patch(
        "src.reporting.discord.discord_bot.get_monthly_report_summary",
        side_effect=Exception("DB error"),
    )
    def test_handle_monthlyreport_command_sends_error_on_exception(self, _mock):
        from src.reporting.discord.discord_bot import handle_monthlyreport_command

        message = MagicMock()
        message.content = "/monthlyreport"
        message.channel = MagicMock()
        message.channel.send = AsyncMock()

        asyncio.run(handle_monthlyreport_command(message))

        calls = message.channel.send.call_args_list
        full_text = " ".join(str(c) for c in calls)
        self.assertIn("失敗", full_text)


class TestHandleForecastCommand(unittest.TestCase):
    def _make_message(self):
        message = MagicMock()
        message.channel = MagicMock()
        message.channel.send = AsyncMock()
        return message

    @patch("src.reporting.discord.discord_bot.get_latest_market_prediction_snapshots")
    def test_calls_snapshot_service(self, mock_get):
        from src.reporting.discord.discord_bot import handle_forecast_command

        mock_get.return_value = (None, [])
        asyncio.run(handle_forecast_command(self._make_message()))
        mock_get.assert_called_once()

    @patch("src.reporting.discord.discord_bot.get_latest_market_prediction_snapshots")
    def test_no_result_when_ts_is_none(self, mock_get):
        from src.reporting.discord.discord_bot import handle_forecast_command

        mock_get.return_value = (None, [])
        message = self._make_message()
        asyncio.run(handle_forecast_command(message))
        message.channel.send.assert_called_once()
        self.assertIn("見つかりませんでした", str(message.channel.send.call_args))

    @patch("src.reporting.discord.discord_bot.get_latest_market_prediction_snapshots")
    def test_no_result_when_snapshots_empty(self, mock_get):
        from src.reporting.discord.discord_bot import handle_forecast_command

        mock_get.return_value = ("2026-04-18T09:00:00", [])
        message = self._make_message()
        asyncio.run(handle_forecast_command(message))
        message.channel.send.assert_called_once()
        self.assertIn("見つかりませんでした", str(message.channel.send.call_args))

    @patch("src.reporting.discord.discord_bot.get_latest_market_prediction_snapshots")
    def test_sends_top_and_worst_tables(self, mock_get):
        from src.reporting.discord.discord_bot import handle_forecast_command
        from src.prediction.types import MarketPredictionSnapshot

        snapshot = MarketPredictionSnapshot(
            market="JP",
            top_results=[
                PredictionResult(
                    market="jp",
                    symbol="7203",
                    current_price=1000.0,
                    avg_pred_price=1015.0,
                    diff_ratio=0.015,
                    model_count=2,
                )
            ],
            worst_results=[
                PredictionResult(
                    market="jp",
                    symbol="9984",
                    current_price=5000.0,
                    avg_pred_price=4900.0,
                    diff_ratio=-0.02,
                    model_count=2,
                )
            ],
        )
        mock_get.return_value = ("2026-04-18T09:00:00", [snapshot])
        message = self._make_message()
        asyncio.run(handle_forecast_command(message))
        self.assertGreaterEqual(message.channel.send.call_count, 2)
        all_text = " ".join(str(c) for c in message.channel.send.call_args_list)
        self.assertIn("7203", all_text)
        self.assertIn("9984", all_text)

    @patch("src.reporting.discord.discord_bot.get_latest_market_prediction_snapshots")
    def test_skips_empty_top_results(self, mock_get):
        from src.reporting.discord.discord_bot import handle_forecast_command
        from src.prediction.types import MarketPredictionSnapshot

        snapshot = MarketPredictionSnapshot(market="JP", top_results=[], worst_results=[])
        mock_get.return_value = ("2026-04-18T09:00:00", [snapshot])
        message = self._make_message()
        asyncio.run(handle_forecast_command(message))
        message.channel.send.assert_not_called()


class TestHandleWatchnextCommand(unittest.TestCase):
    def _make_message(self):
        message = MagicMock()
        message.channel = MagicMock()
        message.channel.send = AsyncMock()
        return message

    @patch("src.reporting.discord.discord_bot.get_watchlist_prediction_view")
    def test_sends_watchlist_table_on_success(self, mock_get):
        from src.reporting.discord.discord_bot import handle_watchnext_command
        from src.watchlist.types import WatchlistPredictionRow, WatchlistPredictionView

        view = WatchlistPredictionView(
            rows=[
                WatchlistPredictionRow(
                    symbol="7203",
                    current_price=1000.0,
                    avg_pred_price=1015.0,
                    diff_ratio=0.015,
                )
            ]
        )
        mock_get.return_value = view
        message = self._make_message()
        asyncio.run(handle_watchnext_command(message))
        message.channel.send.assert_called()
        self.assertIn("7203", str(message.channel.send.call_args))

    @patch("src.reporting.discord.discord_bot.get_watchlist_prediction_view")
    def test_sends_error_text_on_failure(self, mock_get):
        from src.reporting.discord.discord_bot import handle_watchnext_command
        from src.watchlist.types import WatchlistPredictionView

        view = WatchlistPredictionView(error_message="DB接続エラー")
        mock_get.return_value = view
        message = self._make_message()
        asyncio.run(handle_watchnext_command(message))
        message.channel.send.assert_called()
        self.assertIn("DB接続エラー", str(message.channel.send.call_args))

    @patch("src.reporting.discord.discord_bot.get_watchlist_prediction_view")
    def test_sends_default_error_when_error_message_empty(self, mock_get):
        from src.reporting.discord.discord_bot import handle_watchnext_command
        from src.watchlist.types import WatchlistPredictionView

        view = WatchlistPredictionView(error_message="")
        mock_get.return_value = view
        message = self._make_message()
        asyncio.run(handle_watchnext_command(message))
        message.channel.send.assert_called()
        self.assertIn("エラー", str(message.channel.send.call_args))


if __name__ == "__main__":
    unittest.main()
