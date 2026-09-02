"""ユニットテスト: src.trading.allocation_strategy.equity"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.trading.allocation_strategy.equity import (
    build_allocation_equity_series,
    get_allocation_equity_curve,
)
from src.trading.allocation_strategy.types import AllocationSnapshot


def _snapshot(
    id_,
    executed_at,
    action,
    tqqq_qty_after,
    shy_qty_after,
    cash_after,
):
    return AllocationSnapshot(
        id=id_,
        executed_at=executed_at,
        action=action,
        tqqq_price=0.0,
        shy_price=0.0,
        tqqq_qty_before=0.0,
        shy_qty_before=0.0,
        cash_before=0.0,
        tqqq_qty_after=tqqq_qty_after,
        shy_qty_after=shy_qty_after,
        cash_after=cash_after,
    )


def _index(n=10, start="2026-01-05"):
    return pd.bdate_range(start=start, periods=n)


def _flat_prices(price, index):
    return pd.Series(price, index=index)


class TestBuildAllocationEquitySeries(unittest.TestCase):
    def test_empty_snapshots_returns_empty(self):
        idx = _index()
        result = build_allocation_equity_series(
            [], _flat_prices(100.0, idx), _flat_prices(80.0, idx), idx
        )
        self.assertTrue(result.empty)

    def test_empty_index_returns_empty(self):
        snap = [_snapshot(1, datetime(2026, 1, 1), "initial", 100.0, 50.0, 10.0)]
        result = build_allocation_equity_series(
            snap, pd.Series(dtype=float), pd.Series(dtype=float), pd.DatetimeIndex([])
        )
        self.assertTrue(result.empty)

    def test_single_snapshot_marks_to_market(self):
        """スナップショット1件のみ: 全日で同じ保有数量×終値+現金になること"""
        idx = _index()
        snap = [_snapshot(1, idx[0], "initial", 100.0, 50.0, 10.0)]
        tqqq = _flat_prices(80.0, idx)
        shy = _flat_prices(85.0, idx)

        equity = build_allocation_equity_series(snap, tqqq, shy, idx)

        expected = 100.0 * 80.0 + 50.0 * 85.0 + 10.0
        for v in equity:
            self.assertAlmostEqual(v, expected)

    def test_price_appreciation_increases_equity(self):
        idx = _index()
        snap = [_snapshot(1, idx[0], "initial", 100.0, 0.0, 0.0)]
        tqqq = pd.Series(80.0, index=idx)
        tqqq[tqqq.index >= idx[5]] = 90.0
        shy = _flat_prices(0.0, idx)

        equity = build_allocation_equity_series(snap, tqqq, shy, idx)

        self.assertAlmostEqual(equity.iloc[0], 8_000.0)
        self.assertAlmostEqual(equity.iloc[-1], 9_000.0)

    def test_rebalance_snapshot_changes_holdings_from_its_date(self):
        """2件目(rebalance)以降は新しい保有数量が適用されること"""
        idx = _index(n=10)
        snap = [
            _snapshot(1, idx[0], "initial", 100.0, 50.0, 0.0),
            _snapshot(2, idx[5], "rebalance", 120.0, 30.0, 0.0),
        ]
        tqqq = _flat_prices(80.0, idx)
        shy = _flat_prices(85.0, idx)

        equity = build_allocation_equity_series(snap, tqqq, shy, idx)

        before = 100.0 * 80.0 + 50.0 * 85.0
        after = 120.0 * 80.0 + 30.0 * 85.0
        self.assertAlmostEqual(equity.iloc[4], before)
        self.assertAlmostEqual(equity.iloc[5], after)
        self.assertAlmostEqual(equity.iloc[-1], after)

    def test_missing_price_is_forward_filled(self):
        idx = _index(n=5)
        snap = [_snapshot(1, idx[0], "initial", 10.0, 0.0, 0.0)]
        tqqq = pd.Series([100.0, None, None, None, None], index=idx)
        shy = _flat_prices(0.0, idx)

        equity = build_allocation_equity_series(snap, tqqq, shy, idx)

        for v in equity:
            self.assertAlmostEqual(v, 1_000.0)


class TestGetAllocationEquityCurve(unittest.TestCase):
    @patch("src.trading.allocation_strategy.equity.list_snapshots")
    def test_no_snapshots_returns_empty_series_without_touching_port(self, mock_list):
        mock_list.return_value = []
        mock_port = MagicMock()

        result = get_allocation_equity_curve(mock_port)

        self.assertTrue(result.empty)
        mock_port.get_stock_data.assert_not_called()

    @patch("src.trading.allocation_strategy.equity.list_snapshots")
    def test_price_fetch_failure_returns_empty_series(self, mock_list):
        mock_list.return_value = [_snapshot(1, datetime(2026, 1, 1), "initial", 100.0, 50.0, 10.0)]
        mock_port = MagicMock()
        mock_port.get_stock_data.return_value = pd.DataFrame()

        result = get_allocation_equity_curve(mock_port, days=10)

        self.assertTrue(result.empty)

    @patch("src.trading.allocation_strategy.equity.datetime")
    @patch("src.trading.allocation_strategy.equity.list_snapshots")
    def test_uses_port_prices_to_build_series(self, mock_list, mock_datetime):
        idx = _index(n=5, start="2026-01-05")
        mock_datetime.now.return_value = datetime(2026, 1, 12)
        mock_list.return_value = [_snapshot(1, idx[0], "initial", 100.0, 50.0, 10.0)]
        mock_port = MagicMock()
        price_df = pd.DataFrame({"Close": [80.0] * len(idx)}, index=idx)
        mock_port.get_stock_data.return_value = price_df

        result = get_allocation_equity_curve(mock_port, days=10)

        self.assertFalse(result.empty)
        expected = 100.0 * 80.0 + 50.0 * 80.0 + 10.0
        self.assertAlmostEqual(result.iloc[-1], expected)


if __name__ == "__main__":
    unittest.main()
