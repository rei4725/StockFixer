"""ユニットテスト: src.trading.regime_leverage_strategy.repository"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestGetLatestSnapshot(unittest.TestCase):
    def _mock_db(self, row):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = row
        return mock_con

    def test_returns_none_when_no_rows(self):
        from src.trading.regime_leverage_strategy.repository import get_latest_snapshot

        mock_con = self._mock_db(None)
        with patch("src.trading.regime_leverage_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_latest_snapshot()
        self.assertIsNone(result)

    def test_parses_row_into_snapshot(self):
        from src.trading.regime_leverage_strategy.repository import get_latest_snapshot

        executed_at = datetime(2026, 9, 1)
        entry_date = datetime(2026, 9, 1)
        row = (
            3,
            executed_at,
            "entry",
            "regime_entry",
            560.0,
            148.5,
            3500.0,
            entry_date,
            83160.0,
            500.0,
            1000000.0,
            78960.0,
            1005000.0,
            None,
        )
        mock_con = self._mock_db(row)
        with patch("src.trading.regime_leverage_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_latest_snapshot()

        # Verify all 14 fields
        self.assertEqual(result.id, 3)
        self.assertEqual(result.executed_at, executed_at)
        self.assertEqual(result.action, "entry")
        self.assertEqual(result.reason, "regime_entry")
        self.assertEqual(result.spy_price_usd, 560.0)
        self.assertEqual(result.usdjpy_rate, 148.5)
        self.assertEqual(result.shares, 3500.0)
        self.assertEqual(result.entry_date, entry_date)
        self.assertEqual(result.entry_price_jpy, 83160.0)
        self.assertEqual(result.entry_commission_jpy, 500.0)
        self.assertEqual(result.equity_at_entry_jpy, 1000000.0)
        self.assertEqual(result.stop_price_jpy, 78960.0)
        self.assertEqual(result.equity_now_jpy, 1005000.0)
        self.assertIsNone(result.maintenance_ratio)


class TestInsertSnapshot(unittest.TestCase):
    def test_executes_insert_with_expected_params(self):
        from src.trading.regime_leverage_strategy.repository import insert_snapshot
        from src.trading.regime_leverage_strategy.types import RegimeLeverageDecision

        entry_date = datetime(2026, 9, 1)
        decision = RegimeLeverageDecision(
            action="entry",
            reason="regime_entry",
            spy_price_usd=560.0,
            usdjpy_rate=148.5,
            shares=3500.0,
            entry_date=entry_date,
            entry_price_jpy=83160.0,
            entry_commission_jpy=500.0,
            equity_at_entry_jpy=1000000.0,
            stop_price_jpy=78960.0,
            equity_now_jpy=1005000.0,
            maintenance_ratio=None,
        )
        mock_con = MagicMock()
        with patch("src.trading.regime_leverage_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            insert_snapshot(decision)

        mock_con.execute.assert_called_once()
        sql_call = mock_con.execute.call_args
        self.assertIn("INSERT INTO regime_leverage_log", sql_call[0][0])
        # Verify all 12 parameters match decision fields in exact order
        expected_params = [
            decision.action,
            decision.reason,
            decision.spy_price_usd,
            decision.usdjpy_rate,
            decision.shares,
            decision.entry_date,
            decision.entry_price_jpy,
            decision.entry_commission_jpy,
            decision.equity_at_entry_jpy,
            decision.stop_price_jpy,
            decision.equity_now_jpy,
            decision.maintenance_ratio,
        ]
        self.assertEqual(sql_call[0][1], expected_params)


if __name__ == "__main__":
    unittest.main()
