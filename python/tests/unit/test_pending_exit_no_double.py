"""ユニットテスト: 約定待ちの決済がある銘柄に2本目の決済を出さない（#1117）

前の日の 16:30 のルールの道が出した約定待ちの SELL を見ずに、8:50 の自動発注の
予想の売りと SL/TP の売りが同じ銘柄へ2本目の SELL を出していた。空売りの返済
（SHORT_COVER）も同じ形で重ねない。約定待ちの無い銘柄は今までどおり決済する。
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.infrastructure.in_memory import (
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import _check_sl_tp_exits, run_daily_orders
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection

_PREDICTIONS = InMemoryPredictionRepository()


def _broker(positions=None, short_positions=None) -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = positions or []
    broker.get_short_positions.return_value = short_positions or []
    broker.send_order.return_value = {"order_id": "ord-1117", "status": "pending"}
    return broker


def _stats() -> dict:
    return {
        "buy_orders": 0,
        "sell_orders": 0,
        "short_orders": 0,
        "skipped": 0,
        "skipped_min_change": 0,
        "errors": 0,
        "trading_stopped": False,
        "stop_reason": None,
        "reason_code": None,
        "daily_loss": 0.0,
        "daily_loss_limit": None,
        "total_turnover": 0.0,
        "correlation_blocked": False,
        "enc": 0.0,
        "avg_correlation": 0.0,
        "n_held_symbols": 0,
        "held_symbols_list": [],
    }


def _row(symbol: str, diff_ratio: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "current_price": 800.0,
        "diff_ratio": diff_ratio,
        "confidence_ratio": 1.0,
    }


def _long(symbol: str) -> dict:
    return {"symbol": symbol, "qty": 100, "avg_price": 1000.0, "current_price": 800.0}


def _short(symbol: str) -> dict:
    return {"symbol": symbol, "qty": 100, "avg_short_price": 1000.0, "current_price": 1200.0}


def _sides(broker: MagicMock, side: OrderSide) -> list[str]:
    return [c.args[0] for c in broker.send_order.call_args_list if c.args[1] == side]


class _PendingOrdersCase(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
        self.addCleanup(self._clear)

    def _clear(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")

    def _pending(self, symbol: str, side: OrderSide) -> None:
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_orders "
                "(order_id, market, symbol, side, qty, price, order_type, status, created_at) "
                "VALUES (%s, 'jp', %s, %s, 100, 0.0, %s, 'pending', CURRENT_TIMESTAMP)",
                [f"p-{symbol}-{int(side)}", symbol, int(side), int(OrderType.MARKET)],
            )


class TestSlTpSkipsPendingExit(_PendingOrdersCase):
    def _run(self, broker):
        with patch("src.trading.execution.sl_tp.RiskManager") as risk:
            risk.return_value.check_sl_tp.return_value = (True, False, "SL")
            with patch("src.trading.execution.sl_tp._record_order"):
                with self.assertLogs("src.trading.execution", level="INFO") as logs:
                    triggered = _check_sl_tp_exits(
                        broker, "jp", "paper", MagicMock(), _stats(), InMemoryTradeDiffSink()
                    )
        return triggered, logs

    def test_pending_sell_gets_no_second_sl_tp_sell(self):
        self._pending("9000", OrderSide.SELL)
        broker = _broker(positions=[_long("9000"), _long("7203")])

        triggered, logs = self._run(broker)

        self.assertEqual(_sides(broker, OrderSide.SELL), ["7203"])
        self.assertNotIn("9000", triggered)
        self.assertTrue(any("9000" in line for line in logs.output))

    def test_pending_short_cover_gets_no_second_sl_tp_cover(self):
        self._pending("6758", OrderSide.SHORT_COVER)
        broker = _broker(short_positions=[_short("6758"), _short("9984")])

        self._run(broker)

        self.assertEqual(_sides(broker, OrderSide.SHORT_COVER), ["9984"])


class TestRunnerSkipsPendingExit(_PendingOrdersCase):
    def _run(self, broker, predictions, extra_patches=()):
        gate = TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=None,
        )
        patches = [
            patch.object(_PREDICTIONS, "get_latest_by_market", return_value=predictions),
            patch("src.trading.execution.runner._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=gate,
            ),
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=0),
            patch("src.trading.execution.sl_tp.RiskManager"),
            patch(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            ),
            *extra_patches,
        ]
        for p in patches:
            started = p.start()
            self.addCleanup(p.stop)
            if p.attribute == "RiskManager":
                started.return_value.check_sl_tp.return_value = (False, False, "")
        return run_daily_orders(
            broker,
            order_run_sink=InMemoryOrderRunSink(),
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=_PREDICTIONS,
            market="jp",
            mode="paper",
        )

    def test_pending_sell_gets_no_second_prediction_sell(self):
        self._pending("9000", OrderSide.SELL)
        broker = _broker(positions=[_long("9000"), _long("7203")])
        predictions = pd.DataFrame([_row("9000", -0.05), _row("7203", -0.05)])

        stats = self._run(broker, predictions)

        self.assertEqual(_sides(broker, OrderSide.SELL), ["7203"])
        self.assertEqual(stats["sell_orders"], 1)

    def test_pending_short_cover_gets_no_second_prediction_cover(self):
        self._pending("6758", OrderSide.SHORT_COVER)
        broker = _broker(short_positions=[_short("6758"), _short("9984")])
        predictions = pd.DataFrame([_row("6758", 0.05), _row("9984", 0.05)])

        self._run(
            broker,
            predictions,
            (patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True),),
        )

        self.assertEqual(_sides(broker, OrderSide.SHORT_COVER), ["9984"])


if __name__ == "__main__":
    unittest.main()
