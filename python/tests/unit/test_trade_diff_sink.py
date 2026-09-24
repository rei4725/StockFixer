"""ユニットテスト: TradeDiffRecord / TradeDiffSink / InMemoryTradeDiffSink。"""

import unittest
from datetime import datetime

from src.domain.ports import TradeDiffSink
from src.domain.types import TradeDiffRecord
from src.infrastructure.in_memory import InMemoryTradeDiffSink


class TestTradeDiffRecord(unittest.TestCase):
    def test_required_fields_and_defaults(self):
        rec = TradeDiffRecord(
            market="jp",
            symbol="7203",
            predicted_at="2026-09-23T00:00:00",
            side=1,
            signal_price=1000.0,
            mode="paper",
            order_id="ord-1",
        )
        self.assertEqual(rec.market, "jp")
        self.assertEqual(rec.symbol, "7203")
        self.assertEqual(rec.predicted_at, "2026-09-23T00:00:00")
        self.assertEqual(rec.side, 1)
        self.assertEqual(rec.signal_price, 1000.0)
        self.assertEqual(rec.mode, "paper")
        self.assertEqual(rec.order_id, "ord-1")
        self.assertIsNone(rec.actual_price)
        self.assertIsNone(rec.checked_at)
        self.assertEqual(rec.order_session, "open")
        self.assertIsNone(rec.split_ratio)

    def test_optional_fields_are_settable(self):
        now = datetime(2026, 9, 23, 9, 0, 0)
        rec = TradeDiffRecord(
            market="jp",
            symbol="7203",
            predicted_at="2026-09-23T00:00:00",
            side=1,
            signal_price=1000.0,
            mode="live",
            order_id="ord-2",
            actual_price=1005.0,
            checked_at=now,
            order_session="close",
            split_ratio=0.5,
        )
        self.assertEqual(rec.actual_price, 1005.0)
        self.assertEqual(rec.checked_at, now)
        self.assertEqual(rec.order_session, "close")
        self.assertEqual(rec.split_ratio, 0.5)


class TestInMemoryTradeDiffSink(unittest.TestCase):
    def _record(self, symbol: str = "7203") -> TradeDiffRecord:
        return TradeDiffRecord(
            market="jp",
            symbol=symbol,
            predicted_at="2026-09-23T00:00:00",
            side=1,
            signal_price=1000.0,
            mode="paper",
            order_id=f"ord-{symbol}",
        )

    def test_implements_port(self):
        self.assertIsInstance(InMemoryTradeDiffSink(), TradeDiffSink)

    def test_starts_empty(self):
        self.assertEqual(InMemoryTradeDiffSink().recorded, [])

    def test_record_appends_in_order(self):
        sink = InMemoryTradeDiffSink()
        sink.record(self._record("7203"))
        sink.record(self._record("6758"))
        self.assertEqual([r.symbol for r in sink.recorded], ["7203", "6758"])

    def test_port_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            TradeDiffSink()  # type: ignore[abstract]


if __name__ == "__main__":
    unittest.main()
