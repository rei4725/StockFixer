"""ユニットテスト: 空売り中の銘柄に新規の買いを出さない（#1348）

空売りの保有（約定済み・約定待ちの SHORT の仮登録のどちらも）がある銘柄に買いシグナルが
出た日は、新規の BUY を出さず、約定済みの分だけ SHORT_COVER を出す。
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
from src.trading.execution import run_daily_orders
from src.trading.types import CorrelationGateResult, TradingGateStatus

_GATE = TradingGateStatus(
    is_allowed=True,
    stop_active=False,
    reason_code=None,
    reason=None,
    daily_loss=0.0,
    daily_loss_limit=None,
)
_CORR = CorrelationGateResult(
    is_allowed=True, enc=5.0, enc_threshold=2.0, avg_correlation=0.1, n_symbols=1
)


def _row(symbol: str, diff_ratio: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "current_price": 1000.0,
        "diff_ratio": diff_ratio,
        "confidence_ratio": 1.0,
    }


# 6758 は約定済みの空売り、6501 は約定待ちの SHORT だけ、7000 は空売りの無い新規の買い
_PREDICTIONS = pd.DataFrame([_row("6758", 0.05), _row("6501", 0.05), _row("7000", 0.05)])


def _broker() -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = []
    broker.get_short_positions.return_value = [
        {"symbol": "6758", "qty": 100, "avg_short_price": 1000.0, "current_price": 1000.0},
        {
            "symbol": "6501",
            "qty": 100,
            "pending_qty": 100,
            "avg_short_price": 1000.0,
            "current_price": 1000.0,
        },
    ]
    broker.send_order.return_value = {"order_id": "ord-1348", "status": "pending"}
    return broker


class TestEntrySkipsShortHeldBuy(unittest.TestCase):
    def setUp(self):
        self.broker = _broker()
        self.repo = InMemoryPredictionRepository()
        patches = {
            "repo": patch.object(
                self.repo, "get_latest_by_market", return_value=_PREDICTIONS.copy()
            ),
            "record": patch("src.trading.execution.runner._record_order"),
            "params": patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            "peak": patch("src.trading.execution.RiskManager.update_peak_balance"),
            "gate": patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate", return_value=_GATE
            ),
            "size": patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
            "corr": patch(
                "src.trading.execution.runner.evaluate_correlation_gate", return_value=_CORR
            ),
            "sl_tp": patch("src.trading.execution.sl_tp.RiskManager"),
            "sector": patch(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            ),
            "short": patch("src.trading.execution.runner.ENABLE_SHORT_SIDE", True),
            "pending": patch("src.trading.execution.runner._load_pending_exits"),
            "pending_buy": patch(
                "src.trading.execution.runner._get_pending_buy_symbols", return_value=set()
            ),
            "exit_model": patch("src.trading.execution.runner._load_exit_model", return_value=None),
        }
        for p in patches.values():
            mock = p.start()
            self.addCleanup(p.stop)
            if p is patches["sl_tp"]:
                mock.return_value.check_sl_tp.return_value = (False, False, "")
            if p is patches["pending"]:
                mock.return_value.has.return_value = False

    def _sent(self) -> list[tuple[str, OrderSide]]:
        return [(c.args[0], c.args[1]) for c in self.broker.send_order.call_args_list]

    def test_short_held_symbols_get_cover_not_buy(self):
        run_daily_orders(
            self.broker,
            order_run_sink=InMemoryOrderRunSink(),
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=self.repo,
            market="jp",
            mode="paper",
        )
        self.assertEqual(self._sent(), [("7000", OrderSide.BUY), ("6758", OrderSide.SHORT_COVER)])


if __name__ == "__main__":
    unittest.main()
