"""ユニットテスト: src.trading.allocation_strategy.service"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.trading.allocation_strategy.types import AllocationSnapshot


class TestRunAllocationRebalance(unittest.TestCase):
    def _patch_settings(self):
        return (
            patch("config.settings.ALLOCATION_STRATEGY_TQQQ_SYMBOL", "TQQQ"),
            patch("config.settings.ALLOCATION_STRATEGY_BOND_SYMBOL", "SHY"),
            patch("config.settings.ALLOCATION_STRATEGY_TQQQ_RATIO", 0.8),
            patch("config.settings.ALLOCATION_STRATEGY_INITIAL_CAPITAL", 100_000.0),
            patch("config.settings.ALLOCATION_STRATEGY_REBALANCE_YEARS", 2),
        )

    def _start_settings_patchers(self):
        patchers = self._patch_settings()
        for p in patchers:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patchers])

    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_creates_initial_position_when_no_prior_state(
        self, mock_datetime, mock_get_latest, mock_insert
    ):
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        mock_get_latest.return_value = None
        mock_adapter = MagicMock()
        mock_adapter.get_latest_price.side_effect = lambda symbol: {
            "TQQQ": 100.0,
            "SHY": 50.0,
        }[symbol]
        self._start_settings_patchers()

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance(mock_adapter)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.action, "initial")
        self.assertAlmostEqual(outcome.tqqq_qty_after, 800.0)
        self.assertAlmostEqual(outcome.shy_qty_after, 400.0)
        self.assertAlmostEqual(outcome.cash_after, 0.0)
        mock_insert.assert_called_once()

    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_skips_when_already_run_today(self, mock_datetime, mock_get_latest, mock_insert):
        mock_datetime.now.return_value = datetime(2026, 1, 1, 15, 0, 0)
        mock_get_latest.return_value = AllocationSnapshot(
            id=1,
            executed_at=datetime(2026, 1, 1, 10, 0, 0),
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
        mock_adapter = MagicMock()

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance(mock_adapter)

        self.assertIsNone(outcome)
        mock_insert.assert_not_called()
        mock_adapter.get_latest_price.assert_not_called()

    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_skips_when_rebalance_not_yet_due(self, mock_datetime, mock_get_latest, mock_insert):
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        mock_get_latest.return_value = AllocationSnapshot(
            id=1,
            executed_at=datetime(2025, 6, 1),
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
        mock_adapter = MagicMock()
        self._start_settings_patchers()

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance(mock_adapter)

        self.assertIsNone(outcome)
        mock_insert.assert_not_called()
        mock_adapter.get_latest_price.assert_not_called()

    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_rebalances_when_due(self, mock_datetime, mock_get_latest, mock_insert):
        mock_datetime.now.return_value = datetime(2026, 9, 1)
        mock_get_latest.return_value = AllocationSnapshot(
            id=1,
            executed_at=datetime(2024, 1, 1),
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
        mock_adapter = MagicMock()
        mock_adapter.get_latest_price.side_effect = lambda symbol: {
            "TQQQ": 120.0,
            "SHY": 50.0,
        }[symbol]
        self._start_settings_patchers()

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance(mock_adapter)

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.action, "rebalance")
        self.assertAlmostEqual(outcome.tqqq_qty_before, 800.0)
        self.assertAlmostEqual(outcome.shy_qty_before, 400.0)
        self.assertAlmostEqual(outcome.tqqq_qty_after, 773.333333, places=4)
        self.assertAlmostEqual(outcome.shy_qty_after, 464.0, places=4)
        self.assertAlmostEqual(outcome.cash_after, 0.0, places=4)
        mock_insert.assert_called_once()

    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_aborts_when_price_fetch_fails(self, mock_datetime, mock_get_latest, mock_insert):
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        mock_get_latest.return_value = None
        mock_adapter = MagicMock()
        mock_adapter.get_latest_price.side_effect = lambda symbol: {
            "TQQQ": 100.0,
            "SHY": 0.0,
        }[symbol]
        self._start_settings_patchers()

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance(mock_adapter)

        self.assertIsNone(outcome)
        mock_insert.assert_not_called()

    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_aborts_when_ratio_is_invalid(self, mock_datetime, mock_get_latest, mock_insert):
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        mock_get_latest.return_value = None
        mock_adapter = MagicMock()

        settings_patches = [
            patch("config.settings.ALLOCATION_STRATEGY_TQQQ_RATIO", 1.5),
            patch("config.settings.ALLOCATION_STRATEGY_TQQQ_SYMBOL", "TQQQ"),
            patch("config.settings.ALLOCATION_STRATEGY_BOND_SYMBOL", "SHY"),
            patch("config.settings.ALLOCATION_STRATEGY_INITIAL_CAPITAL", 100_000.0),
            patch("config.settings.ALLOCATION_STRATEGY_REBALANCE_YEARS", 2),
        ]
        for p in settings_patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in settings_patches])

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance(mock_adapter)

        self.assertIsNone(outcome)
        mock_insert.assert_not_called()
        mock_adapter.get_latest_price.assert_not_called()

    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_aborts_when_symbols_are_identical(self, mock_datetime, mock_get_latest, mock_insert):
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        mock_get_latest.return_value = None
        mock_adapter = MagicMock()

        settings_patches = [
            patch("config.settings.ALLOCATION_STRATEGY_TQQQ_RATIO", 0.8),
            patch("config.settings.ALLOCATION_STRATEGY_TQQQ_SYMBOL", "TQQQ"),
            patch("config.settings.ALLOCATION_STRATEGY_BOND_SYMBOL", "TQQQ"),
            patch("config.settings.ALLOCATION_STRATEGY_INITIAL_CAPITAL", 100_000.0),
            patch("config.settings.ALLOCATION_STRATEGY_REBALANCE_YEARS", 2),
        ]
        for p in settings_patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in settings_patches])

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance(mock_adapter)

        self.assertIsNone(outcome)
        mock_insert.assert_not_called()
        mock_adapter.get_latest_price.assert_not_called()


if __name__ == "__main__":
    unittest.main()
