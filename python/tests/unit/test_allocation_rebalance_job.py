from unittest.mock import MagicMock, patch

from src.orchestration.jobs.periodic import run_allocation_rebalance_job


class TestRunAllocationRebalanceJob:
    @patch("src.reporting.discord.discord_utils.send_allocation_rebalance_report")
    @patch("src.trading.allocation_strategy.service.run_allocation_rebalance")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_runs_and_notifies_when_outcome_present(self, mock_adapter_cls, mock_run, mock_notify):
        from src.trading.allocation_strategy.types import RebalanceOutcome

        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_run.return_value = RebalanceOutcome(
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

        run_allocation_rebalance_job()

        mock_run.assert_called_once_with(mock_adapter)
        mock_notify.assert_called_once_with(
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

    @patch("src.reporting.discord.discord_utils.send_allocation_rebalance_report")
    @patch("src.trading.allocation_strategy.service.run_allocation_rebalance", return_value=None)
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_does_not_notify_when_service_returns_none(
        self, mock_adapter_cls, mock_run, mock_notify
    ):
        run_allocation_rebalance_job()

        mock_run.assert_called_once()
        mock_notify.assert_not_called()

    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    @patch(
        "src.trading.allocation_strategy.service.run_allocation_rebalance",
        side_effect=Exception("network error"),
    )
    def test_does_not_raise_on_service_failure(self, mock_run, mock_adapter_cls):
        run_allocation_rebalance_job()  # 例外を送出しないことを確認

    @patch(
        "src.reporting.discord.discord_utils.send_allocation_rebalance_report",
        side_effect=Exception("discord down"),
    )
    @patch("src.trading.allocation_strategy.service.run_allocation_rebalance")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_does_not_raise_on_notification_failure(self, mock_adapter_cls, mock_run, mock_notify):
        from src.trading.allocation_strategy.types import RebalanceOutcome

        mock_run.return_value = RebalanceOutcome(
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

        run_allocation_rebalance_job()  # 例外を送出しないことを確認
