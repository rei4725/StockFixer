"""ユニットテスト: OrderRunSummary 値オブジェクト。"""

import dataclasses
import unittest

from src.domain.types import OrderRunSummary


class TestOrderRunSummary(unittest.TestCase):
    def _sample(self) -> OrderRunSummary:
        return OrderRunSummary(
            run_id="abc123456789",
            market="jp",
            mode="paper",
            buy_orders=3,
            sell_orders=1,
            short_orders=0,
            skipped=2,
            skipped_min_change=1,
            total_turnover=1234567.0,
            min_change_ratio=0.005,
        )

    def test_holds_all_fields(self):
        s = self._sample()
        self.assertEqual(s.run_id, "abc123456789")
        self.assertEqual(s.market, "jp")
        self.assertEqual(s.mode, "paper")
        self.assertEqual(s.buy_orders, 3)
        self.assertEqual(s.sell_orders, 1)
        self.assertEqual(s.short_orders, 0)
        self.assertEqual(s.skipped, 2)
        self.assertEqual(s.skipped_min_change, 1)
        self.assertEqual(s.total_turnover, 1234567.0)
        self.assertEqual(s.min_change_ratio, 0.005)

    def test_is_frozen(self):
        s = self._sample()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.buy_orders = 99  # type: ignore[misc]
