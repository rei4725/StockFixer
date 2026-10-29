"""8:50 の回で期限を過ぎた保有の成行の決済が、売りシグナルの指値の SELL に止められないことのテスト（#1388）。

run_daily_auto_order を、実の run_daily_orders と実のホライズン決済で流す。ブローカーの
send_order は paper_orders に約定待ちの行を足し、約定待ちの読み取りは tests/unit/conftest.py の
実 Postgres を使う。
"""

import os
import unittest
from contextlib import ExitStack
from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd

from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection

_SYMBOL = "9000"


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market": "jp",
                "symbol": _SYMBOL,
                "current_price": 800.0,
                "diff_ratio": -0.02,
                "confidence_ratio": 1.0,
                "predicted_at": "2026-10-30T08:00:00",
            }
        ]
    )


def _attach(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["base_threshold"] = 0.001
    result["threshold_scale"] = 1.0
    result["effective_buy_threshold"] = 0.001
    result["effective_sell_threshold"] = -0.001
    return result


def _score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["multi_horizon_score"] = result["diff_ratio"]
    return result


class TestHorizonExitBeforeLimitSell(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM paper_orders")
        self._seq = 0
        self.broker = MagicMock(spec=BrokerBase)
        self.broker.broker_name = "paper"
        self.broker.get_balance.return_value = 1_000_000.0
        self.broker.get_positions.return_value = [
            {"symbol": _SYMBOL, "qty": 100, "avg_price": 1000.0, "current_price": 800.0}
        ]
        self.broker.send_order.side_effect = self._send_order
        self.sl_tp_risk = MagicMock()
        self.sl_tp_risk.check_sl_tp.return_value = (False, False, "")

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {"AUTO_TRADE_MODE": "paper"}))
        stack.enter_context(
            patch("src.trading.brokers.account.factory.make_order_broker", return_value=self.broker)
        )
        repo = stack.enter_context(
            patch(
                "src.infrastructure.persistence.prediction_result_repository."
                "PostgresPredictionResultRepository"
            )
        )
        repo.return_value.get_latest_by_market.return_value = _predictions()
        for target in (
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter",
            "src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter",
            "src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink",
            "src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink",
            "src.reporting.discord.discord_utils.send_daily_order_completion",
            "src.orchestration.jobs.daily.record_loss_limit_streak",
            "src.trading.execution.runner._record_order",
            "src.trading.execution.sl_tp._record_order",
        ):
            stack.enter_context(patch(target))
        # 出来高が少ないか値幅が広い銘柄として、売りシグナルの SELL は指値で出る
        self.choose = stack.enter_context(
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.LIMIT, 790.0, "wide_range=2.178%", "open"),
            )
        )
        gate = TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=None,
        )
        for target, kwargs in (
            ("src.trading.execution.RiskManager.evaluate_trading_gate", {"return_value": gate}),
            ("src.trading.execution.RiskManager.calc_position_size", {"return_value": 100}),
            (
                "src.trading.risk_manager.RiskManager._get_daily_realized_loss",
                {"return_value": 0.0},
            ),
            ("src.trading.risk_manager.RiskManager._get_consecutive_losses", {"return_value": 0}),
            (
                "src.trading.execution.selection.get_symbol_sector",
                {"side_effect": lambda market, symbol: f"UNKNOWN:{market}:{symbol}"},
            ),
            ("src.trading.execution.runner._attach_dynamic_thresholds", {"side_effect": _attach}),
            (
                "src.trading.execution.runner.apply_multi_horizon_score_column",
                {"side_effect": _score},
            ),
            ("src.trading.execution.runner._load_exit_model", {"return_value": None}),
            ("src.trading.execution.sl_tp._sl_tp_risk", {"return_value": self.sl_tp_risk}),
        ):
            stack.enter_context(patch(target, **kwargs))

    def _send_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict[str, Any]:
        self._seq += 1
        order_id = f"ORD-SEND-{self._seq}"
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_orders (order_id, symbol, side, qty, order_type, status) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [order_id, symbol, int(side), qty, int(order_type), "pending"],
            )
        return {"order_id": order_id, "status": "pending"}

    def _insert_buy(self, target_exit_date: str) -> None:
        with _db_connection() as con:
            con.execute(
                "INSERT INTO paper_orders "
                "(order_id, symbol, side, qty, order_type, status, target_exit_date, filled_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    "ORD-BUY",
                    _SYMBOL,
                    int(OrderSide.BUY),
                    100,
                    int(OrderType.MARKET),
                    "filled",
                    target_exit_date,
                    f"{(date.today() - timedelta(days=5)).isoformat()} 00:05:00",
                ],
            )

    def _pending_sell_types(self) -> list[int]:
        with _db_connection() as con:
            rows = con.execute(
                "SELECT order_type FROM paper_orders "
                "WHERE symbol = %s AND side = %s AND status = 'pending' ORDER BY order_id",
                [_SYMBOL, int(OrderSide.SELL)],
            ).fetchall()
        return [int(r[0]) for r in rows]

    def _run(self) -> None:
        from src.orchestration.jobs import daily

        daily.run_daily_auto_order()

    def test_overdue_long_exits_at_market_despite_limit_sell_signal(self):
        """期限を過ぎた買い持ちに売りシグナルの指値が出る回でも、SELL は成行の1本だけになる。"""
        self._insert_buy((date.today() - timedelta(days=1)).isoformat())

        self._run()

        self.assertEqual(self._pending_sell_types(), [int(OrderType.MARKET)])

    def test_not_due_long_sell_signal_stays_limit(self):
        """期限を過ぎていない保有の売りシグナルは、今までどおり指値で出る。"""
        self._insert_buy((date.today() + timedelta(days=3)).isoformat())

        self._run()

        self.assertEqual(self._pending_sell_types(), [int(OrderType.LIMIT)])
        self.choose.assert_called_once()

    def test_overdue_long_with_sl_triggered_gets_one_sell(self):
        """期限を過ぎた保有に SL も発動する回でも、同じ銘柄の SELL は1本だけになる。"""
        self._insert_buy((date.today() - timedelta(days=1)).isoformat())
        self.sl_tp_risk.check_sl_tp.return_value = (True, False, "sl")

        self._run()

        self.assertEqual(self._pending_sell_types(), [int(OrderType.MARKET)])


if __name__ == "__main__":
    unittest.main()
