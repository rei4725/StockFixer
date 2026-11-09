"""紙以外のブローカーで、実現損益の数える元が無い・読めないときにゲートが新規を止めるテスト（#995）"""

import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

from src.trading.brokers.base import BrokerBase
from src.trading.risk_manager import MAX_CONSECUTIVE_LOSSES, MAX_DAILY_LOSS_RATE, RiskManager

# 水曜 2026-10-28 08:50 JST = 2026-10-27 23:50 UTC。区切りは 2026-10-26 15:00 UTC
_NOW = datetime(2026, 10, 27, 23, 50, tzinfo=timezone.utc)


class _FakeTradePnlCon:
    """trade_pnl の有無・読めるかどうか・決済の損益の行を作り分ける"""

    def __init__(
        self,
        exists: bool = True,
        broken: bool = False,
        daily_loss_sum: float = 0.0,
        recent_pnls: Optional[list[float]] = None,
    ) -> None:
        self._exists = exists
        self._broken = broken
        self._daily_loss_sum = daily_loss_sum
        self._recent_pnls = recent_pnls or []

    def execute(self, query: str, params: list) -> MagicMock:
        result = MagicMock()
        if "information_schema.tables" in query:
            result.fetchone.return_value = (1 if self._exists else 0,)
            return result
        if "paper_orders" in query:
            raise AssertionError("紙以外のブローカーで paper_orders を読んだ")
        if self._broken:
            raise RuntimeError("trade_pnl を読めない")
        result.fetchone.return_value = (-self._daily_loss_sum,)
        result.fetchall.return_value = [(pnl,) for pnl in self._recent_pnls]
        return result


def _gate(con: _FakeTradePnlCon, balance: float = 1_000_000.0, env: Optional[dict] = None):
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "kabu"
    broker.get_balance.return_value = balance
    broker.get_positions.return_value = []
    broker.get_short_positions.return_value = []
    risk = RiskManager(broker)

    @contextmanager
    def fake_db():
        yield con

    with (
        patch("src.trading.risk_manager._db_connection", new=fake_db),
        patch("src.trading.daily_loss_window._utc_now", return_value=_NOW),
        patch.dict("os.environ", env or {}, clear=False),
    ):
        return risk.evaluate_trading_gate()


class TestNonPaperLossGate(unittest.TestCase):
    def test_missing_trade_pnl_blocks_with_daily_loss_source_code(self):
        status = _gate(_FakeTradePnlCon(exists=False))
        self.assertFalse(status.is_allowed)
        self.assertTrue(status.stop_active)
        self.assertEqual(status.reason_code, "daily_loss_source_unavailable")

    def test_unreadable_trade_pnl_blocks_with_daily_loss_source_code(self):
        status = _gate(_FakeTradePnlCon(broken=True))
        self.assertFalse(status.is_allowed)
        self.assertEqual(status.reason_code, "daily_loss_source_unavailable")

    def test_missing_trade_pnl_blocks_on_consecutive_when_daily_guard_disabled(self):
        status = _gate(_FakeTradePnlCon(exists=False), env={"DISABLE_DAILY_LOSS_GUARD": "1"})
        self.assertFalse(status.is_allowed)
        self.assertTrue(status.stop_active)
        self.assertEqual(status.reason_code, "consecutive_loss_source_unavailable")

    def test_unreadable_trade_pnl_blocks_on_consecutive_when_daily_guard_disabled(self):
        status = _gate(_FakeTradePnlCon(broken=True), env={"DISABLE_DAILY_LOSS_GUARD": "1"})
        self.assertFalse(status.is_allowed)
        self.assertEqual(status.reason_code, "consecutive_loss_source_unavailable")

    def test_daily_loss_over_limit_blocks(self):
        balance = 1_000_000.0
        loss = balance * MAX_DAILY_LOSS_RATE + 1.0
        status = _gate(_FakeTradePnlCon(daily_loss_sum=loss), balance=balance)
        self.assertFalse(status.is_allowed)
        self.assertEqual(status.reason_code, "daily_loss_limit")
        self.assertEqual(status.daily_loss, loss)

    def test_three_recent_losses_block(self):
        pnls = [-100.0] * MAX_CONSECUTIVE_LOSSES
        status = _gate(_FakeTradePnlCon(recent_pnls=pnls))
        self.assertFalse(status.is_allowed)
        self.assertEqual(status.reason_code, "consecutive_losses")
        self.assertEqual(status.consecutive_losses, MAX_CONSECUTIVE_LOSSES)

    def test_readable_trade_pnl_without_losses_allows(self):
        status = _gate(_FakeTradePnlCon(recent_pnls=[50.0, -10.0]))
        self.assertTrue(status.is_allowed)
        self.assertEqual(status.consecutive_losses, 0)


if __name__ == "__main__":
    unittest.main()
