"""ユニットテスト: InMemoryOrderRunSink（テスト用の偽アダプタ）。"""

import unittest

from src.domain.ports import OrderRunSink
from src.domain.types import OrderRunSummary
from src.infrastructure.in_memory import InMemoryOrderRunSink


def _summary(run_id: str = "run-0001") -> OrderRunSummary:
    return OrderRunSummary(
        run_id=run_id,
        market="jp",
        mode="paper",
        buy_orders=1,
        sell_orders=0,
        short_orders=0,
        skipped=0,
        skipped_min_change=0,
        total_turnover=100.0,
        min_change_ratio=0.005,
    )


class TestInMemoryOrderRunSink(unittest.TestCase):
    def test_implements_port(self):
        self.assertIsInstance(InMemoryOrderRunSink(), OrderRunSink)

    def test_records_saved_summaries_in_order(self):
        sink = InMemoryOrderRunSink()
        sink.save(_summary("run-0001"))
        sink.save(_summary("run-0002"))

        self.assertEqual([s.run_id for s in sink.saved], ["run-0001", "run-0002"])

    def test_starts_empty(self):
        self.assertEqual(InMemoryOrderRunSink().saved, [])
