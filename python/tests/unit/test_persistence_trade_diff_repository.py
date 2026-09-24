"""ユニットテスト: PostgresTradeDiffSink（SQL 移設の回帰検証）。

実 DB には接続せず、db_connection を偽物に差し替えて
発行される SQL と引数を検証する。SQL 本文が変わると落ちる。
"""

import unittest
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch

from src.domain.ports import TradeDiffSink
from src.domain.types import TradeDiffRecord
from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConnection:
    """execute() の呼び出しを (sql, params) で記録する偽接続。"""

    def __init__(self, existing_row=None):
        self.calls: list[tuple[str, list]] = []
        self._existing_row = existing_row

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return _FakeCursor(self._existing_row)


def _patched_connection(con):
    @contextmanager
    def _cm():
        yield con

    return _cm


def _record(**overrides) -> TradeDiffRecord:
    base = dict(
        market="jp",
        symbol="7203",
        predicted_at="2026-09-23T00:00:00",
        side=1,
        signal_price=1000.0,
        mode="paper",
        order_id="ord-1",
    )
    base.update(overrides)
    return TradeDiffRecord(**base)  # type: ignore[arg-type]


class TestPostgresTradeDiffSink(unittest.TestCase):
    def test_implements_port(self):
        self.assertIsInstance(PostgresTradeDiffSink(), TradeDiffSink)

    def test_issues_select_delete_insert_in_order(self):
        con = _FakeConnection(existing_row=None)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(_record())

        self.assertEqual(len(con.calls), 3)
        self.assertIn("SELECT", con.calls[0][0])
        self.assertIn("FROM paper_real_diff", con.calls[0][0])
        self.assertTrue(con.calls[1][0].startswith("DELETE FROM paper_real_diff"))
        self.assertIn("INSERT INTO paper_real_diff", con.calls[2][0])

    def test_paper_mode_fills_paper_columns_and_slippage(self):
        con = _FakeConnection(existing_row=None)
        checked = datetime(2026, 9, 23, 9, 0, 0)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(
                _record(mode="paper", actual_price=1010.0, checked_at=checked)
            )

        insert_params = con.calls[2][1]
        # 並び: market, symbol, predicted_at, side, signal_price,
        #       paper_order_id, real_order_id, paper_price, real_price,
        #       paper_slippage, real_slippage, price_diff,
        #       paper_filled_at, real_checked_at, created_at, order_session, split_ratio
        self.assertEqual(insert_params[5], "ord-1")  # paper_order_id
        self.assertIsNone(insert_params[6])  # real_order_id
        self.assertEqual(insert_params[7], 1010.0)  # paper_price
        self.assertAlmostEqual(insert_params[9], 0.01)  # paper_slippage
        self.assertEqual(insert_params[12], checked)  # paper_filled_at
        self.assertEqual(insert_params[15], "open")  # order_session

    def test_live_mode_fills_real_columns(self):
        con = _FakeConnection(existing_row=None)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(_record(mode="live", actual_price=990.0))

        insert_params = con.calls[2][1]
        self.assertIsNone(insert_params[5])  # paper_order_id
        self.assertEqual(insert_params[6], "ord-1")  # real_order_id
        self.assertEqual(insert_params[8], 990.0)  # real_price
        self.assertAlmostEqual(insert_params[10], -0.01)  # real_slippage

    def test_price_diff_computed_when_both_sides_present(self):
        # 既存行に paper_price=1000 があり、今回 live で 1005 を記録する
        existing = (
            1000.0,  # signal_price
            "paper-1",  # paper_order_id
            None,  # real_order_id
            1000.0,  # paper_price
            None,  # real_price
            0.0,  # paper_slippage
            None,  # real_slippage
            datetime(2026, 9, 22, 9, 0, 0),  # paper_filled_at
            None,  # real_checked_at
            datetime(2026, 9, 22, 8, 0, 0),  # created_at
            "open",  # order_session
            1.0,  # split_ratio
        )
        con = _FakeConnection(existing_row=existing)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(_record(mode="live", actual_price=1005.0))

        insert_params = con.calls[2][1]
        self.assertAlmostEqual(insert_params[11], 5.0)  # price_diff = real - paper

    def test_zero_signal_price_yields_none_slippage(self):
        con = _FakeConnection(existing_row=None)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(
                _record(signal_price=0.0, mode="paper", actual_price=1010.0)
            )

        insert_params = con.calls[2][1]
        self.assertIsNone(insert_params[9])  # paper_slippage


if __name__ == "__main__":
    unittest.main()
