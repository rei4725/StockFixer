"""ユニットテスト: src.orchestration.jobs.periodic の純粋ロジック部分"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.orchestration.jobs.periodic import _build_monthly_equity_series


def _series(n=5, value=100.0):
    idx = pd.bdate_range("2026-01-05", periods=n)
    return pd.Series(value, index=idx)


class TestBuildMonthlyEquitySeries(unittest.TestCase):
    def test_paper_trading_label_includes_start_date(self):
        result = _build_monthly_equity_series(_series(), pd.Series(dtype=float), None)
        self.assertIn("Paper Trading (since 2026-01-05)", result)
        self.assertEqual(len(result), 1)

    def test_allocation_bot_label_includes_its_own_start_date(self):
        allocation = pd.Series(50.0, index=pd.bdate_range("2026-03-02", periods=5))
        result = _build_monthly_equity_series(_series(), allocation, None)
        self.assertIn("Allocation Bot (since 2026-03-02)", result)

    def test_excludes_allocation_bot_when_empty(self):
        result = _build_monthly_equity_series(_series(), pd.Series(dtype=float), None)
        self.assertFalse(any(k.startswith("Allocation Bot") for k in result))

    def test_excludes_allocation_bot_when_fewer_than_5_points(self):
        """メインと同じ最低点数基準: 4点以下は運用開始直後のノイズとして除外"""
        allocation = pd.Series(50.0, index=pd.bdate_range("2026-09-01", periods=4))
        result = _build_monthly_equity_series(_series(), allocation, None)
        self.assertFalse(any(k.startswith("Allocation Bot") for k in result))

    def test_includes_benchmark_label_with_start_date_when_present(self):
        result = _build_monthly_equity_series(
            _series(), pd.Series(dtype=float), _series(value=5000.0)
        )
        self.assertIn("S&P 500 (since 2026-01-05)", result)


class TestRunRegimeLeverageWeeklyJob(unittest.TestCase):
    @patch("src.trading.regime_leverage_strategy.service.run_regime_leverage_weekly_check")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_calls_service_with_adapter_and_logs_decision(self, mock_adapter_cls, mock_run):
        from src.orchestration.jobs.periodic import run_regime_leverage_weekly_job
        from src.trading.regime_leverage_strategy.types import RegimeLeverageDecision

        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_run.return_value = RegimeLeverageDecision(
            action="entry",
            reason="regime_entry",
            spy_price_usd=500.0,
            usdjpy_rate=150.0,
            shares=10.0,
            entry_date=None,
            entry_price_jpy=None,
            entry_commission_jpy=None,
            equity_at_entry_jpy=None,
            stop_price_jpy=None,
            equity_now_jpy=1_000_000.0,
            maintenance_ratio=None,
        )

        with self.assertLogs("src.orchestration.jobs.periodic", level="INFO") as logs:
            run_regime_leverage_weekly_job()

        mock_run.assert_called_once_with(mock_adapter)
        self.assertTrue(
            any(
                "action=entry" in message and "reason=regime_entry" in message
                for message in logs.output
            )
        )

    @patch("src.trading.regime_leverage_strategy.service.run_regime_leverage_weekly_check")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_does_not_log_decision_when_none(self, mock_adapter_cls, mock_run):
        """MA200/ATR14がNaNの場合など、serviceがNoneを返した際はaction=...のログを
        出さず(decision.actionへのアクセスでAttributeErrorにならず)正常終了すること。
        """
        from src.orchestration.jobs.periodic import run_regime_leverage_weekly_job

        mock_run.return_value = None

        with self.assertLogs("src.orchestration.jobs.periodic", level="INFO") as logs:
            run_regime_leverage_weekly_job()

        self.assertFalse(any("action=" in message for message in logs.output))

    @patch("src.trading.regime_leverage_strategy.service.run_regime_leverage_weekly_check")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_does_not_raise_on_failure(self, mock_adapter_cls, mock_run):
        from src.orchestration.jobs.periodic import run_regime_leverage_weekly_job

        mock_run.side_effect = Exception("boom")

        run_regime_leverage_weekly_job()  # 例外を吸収してログのみ出すこと

        mock_run.assert_called_once()


class TestRunRegimeLeverageDailyMarginJob(unittest.TestCase):
    @patch("src.trading.regime_leverage_strategy.service.run_regime_leverage_daily_margin_check")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_calls_service_and_logs_when_decision_present(self, mock_adapter_cls, mock_run):
        from src.orchestration.jobs.periodic import run_regime_leverage_daily_margin_job
        from src.trading.regime_leverage_strategy.types import RegimeLeverageDecision

        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_run.return_value = RegimeLeverageDecision(
            action="exit",
            reason="margin_call",
            spy_price_usd=480.0,
            usdjpy_rate=149.0,
            shares=0.0,
            entry_date=None,
            entry_price_jpy=None,
            entry_commission_jpy=None,
            equity_at_entry_jpy=None,
            stop_price_jpy=None,
            equity_now_jpy=900_000.0,
            maintenance_ratio=0.24,
        )

        with self.assertLogs("src.orchestration.jobs.periodic", level="INFO") as logs:
            run_regime_leverage_daily_margin_job()

        mock_run.assert_called_once_with(mock_adapter)
        self.assertTrue(
            any(
                "action=exit" in message and "maintenance_ratio=0.24" in message
                for message in logs.output
            )
        )

    @patch("src.trading.regime_leverage_strategy.service.run_regime_leverage_daily_margin_check")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_does_not_log_decision_when_none(self, mock_adapter_cls, mock_run):
        from src.orchestration.jobs.periodic import run_regime_leverage_daily_margin_job

        mock_run.return_value = None

        with self.assertLogs("src.orchestration.jobs.periodic", level="INFO") as logs:
            run_regime_leverage_daily_margin_job()

        self.assertFalse(any("action=" in message for message in logs.output))

    @patch("src.trading.regime_leverage_strategy.service.run_regime_leverage_daily_margin_check")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_does_not_raise_on_failure(self, mock_adapter_cls, mock_run):
        from src.orchestration.jobs.periodic import run_regime_leverage_daily_margin_job

        mock_run.side_effect = Exception("boom")

        run_regime_leverage_daily_margin_job()  # 例外を吸収してログのみ出すこと

        mock_run.assert_called_once()


class TestRegimeLeverageScheduleConfig(unittest.TestCase):
    """未建玉のためデフォルト無効(auto_schedule: False)であることを検証する
    (allocation_rebalanceと同じ安全ロールアウト手順。手動 --run-now で初回エントリーを
    確認してから True に切り替える運用のため、切り替え忘れ・意図しない有効化を検知する)。
    """

    def test_daily_margin_not_auto_scheduled(self):
        from run_scheduler import SCHEDULE_CONFIG

        self.assertIs(SCHEDULE_CONFIG["regime_leverage_daily_margin"]["auto_schedule"], False)

    def test_weekly_not_auto_scheduled(self):
        from run_scheduler import SCHEDULE_CONFIG

        self.assertIs(SCHEDULE_CONFIG["regime_leverage_weekly"]["auto_schedule"], False)


if __name__ == "__main__":
    unittest.main()
