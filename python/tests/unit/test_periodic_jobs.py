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
    def test_reraises_on_failure_after_error_log(self, mock_adapter_cls, mock_run):
        """例外はERRORログの後に再送出し、スケジューラーのリトライ・Discord通知に乗せる(#860)。"""
        from src.orchestration.jobs.periodic import run_regime_leverage_weekly_job

        mock_run.side_effect = Exception("boom")

        with self.assertLogs("src.orchestration.jobs.periodic", level="ERROR"):
            with self.assertRaises(Exception):
                run_regime_leverage_weekly_job()

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
    def test_reraises_on_failure_after_error_log(self, mock_adapter_cls, mock_run):
        """例外はERRORログの後に再送出し、スケジューラーのリトライ・Discord通知に乗せる(#860)。"""
        from src.orchestration.jobs.periodic import run_regime_leverage_daily_margin_job

        mock_run.side_effect = Exception("boom")

        with self.assertLogs("src.orchestration.jobs.periodic", level="ERROR"):
            with self.assertRaises(Exception):
                run_regime_leverage_daily_margin_job()

        mock_run.assert_called_once()

    @staticmethod
    def _daily_df(nan_low=False):
        import numpy as np

        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(400.0, 500.0, 260), index=idx)
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 2,
                "Low": prices - 2,
                "Close": prices,
                "Volume": 1_000_000,
            },
            index=idx,
        )
        if nan_low:
            df.loc[df.index[-1], "Low"] = np.nan
        return df

    @staticmethod
    def _holding_snapshot():
        from datetime import datetime

        from src.trading.regime_leverage_strategy.types import RegimeLeverageSnapshot

        return RegimeLeverageSnapshot(
            id=1,
            executed_at=datetime(2026, 1, 2),
            action="entry",
            reason="regime_entry",
            spy_price_usd=500.0,
            usdjpy_rate=145.0,
            shares=27.0,
            entry_date=datetime(2026, 1, 2),
            entry_price_jpy=72500.0,
            entry_commission_jpy=0.0,
            equity_at_entry_jpy=1_000_000.0,
            stop_price_jpy=65000.0,
            equity_now_jpy=1_000_000.0,
            maintenance_ratio=None,
        )

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_raises_when_bar_fetch_fails_while_holding(
        self, mock_adapter_cls, mock_latest, mock_insert
    ):
        """保有中に足の取得が例外になった場合、ジョブ関数が例外で終わること(#860)。"""
        from src.orchestration.jobs.periodic import run_regime_leverage_daily_margin_job

        mock_latest.return_value = self._holding_snapshot()
        mock_adapter_cls.return_value.get_stock_data.side_effect = ConnectionError("yf down")

        with self.assertRaises(ConnectionError):
            run_regime_leverage_daily_margin_job()

        mock_insert.assert_not_called()

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_raises_when_holding_and_day_low_is_nan(
        self, mock_adapter_cls, mock_latest, mock_insert
    ):
        """保有中に当日安値がNaNで判定を中止した場合、ジョブ関数が例外で終わること(#860)。"""
        from src.orchestration.jobs.periodic import run_regime_leverage_daily_margin_job

        mock_latest.return_value = self._holding_snapshot()
        port = mock_adapter_cls.return_value
        port.get_stock_data.return_value = self._daily_df(nan_low=True)
        port.get_forex_data.return_value = pd.DataFrame({"Close": [145.0]})

        with self.assertRaises(RuntimeError):
            run_regime_leverage_daily_margin_job()

        mock_insert.assert_not_called()

    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_does_not_raise_when_not_holding(self, mock_adapter_cls, mock_latest, mock_insert):
        """未保有の場合は今までどおり何もせず正常終了すること(#860)。"""
        from src.orchestration.jobs.periodic import run_regime_leverage_daily_margin_job

        mock_latest.return_value = None

        run_regime_leverage_daily_margin_job()

        mock_adapter_cls.return_value.get_stock_data.assert_not_called()
        mock_insert.assert_not_called()


class TestRegimeLeverageScheduleConfig(unittest.TestCase):
    """auto_schedule設定値を検証する(allocation_rebalanceと同じ安全ロールアウト手順)。

    2026-09-06: 手動 --run-now で初回エントリー・日次/週次動作を確認し、
    ユーザーの運用開始判断によりauto_schedule: Trueへ切り替え済み。
    """

    def test_daily_margin_is_auto_scheduled(self):
        from run_scheduler import SCHEDULE_CONFIG

        self.assertIs(SCHEDULE_CONFIG["regime_leverage_daily_margin"]["auto_schedule"], True)

    def test_weekly_is_auto_scheduled(self):
        from run_scheduler import SCHEDULE_CONFIG

        self.assertIs(SCHEDULE_CONFIG["regime_leverage_weekly"]["auto_schedule"], True)


if __name__ == "__main__":
    unittest.main()
