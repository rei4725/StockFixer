"""
ユニットテスト: optimal_params.json が読めない日の閾値・Kelly 入力（#1214）

- 8:50 の道は既定の閾値で予想の売りまで出し、新規の買いは止め、errors を数える
- Claude の発注の道は既定の Kelly 入力で株数を決める
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from config.settings import BUY_THRESHOLD, DEFAULT_STOP_LOSS_PCT
from src.infrastructure.in_memory import (
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import _attach_dynamic_thresholds, run_daily_orders
from src.trading.execution.params import _resolve_kelly_params
from src.trading.types import CorrelationGateResult, TradingGateStatus
from src.utils.optimal_params_loader import OptimalParamsUnreadableError

_LOSS = DEFAULT_STOP_LOSS_PCT + 0.02
_POSITIONS = [
    {"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0 * (1 - _LOSS)},
    {"symbol": "9002", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
]


def _make_broker():
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = _POSITIONS
    broker.get_short_positions.return_value = []
    broker.send_order.return_value = {"order_id": "test-1214", "status": "pending"}
    return broker


def _unreadable_everywhere():
    # get_optimal_params だけを読めなくする（params 側の関数は差し替えない）
    err = OptimalParamsUnreadableError("broken")
    return [
        patch("src.trading.risk_manager.get_optimal_params", side_effect=err),
        patch("src.trading.execution.params.get_optimal_params", side_effect=err),
    ]


def _predictions():
    return pd.DataFrame(
        [
            {
                "market": "jp",
                "symbol": "9002",
                "predicted_at": "20261026_083000",
                "current_price": 1000.0,
                "diff_ratio": -0.03,
                "confidence_ratio": 1.0,
            },
            {
                "market": "jp",
                "symbol": "6501",
                "predicted_at": "20261026_083000",
                "current_price": 1000.0,
                "diff_ratio": 0.03,
                "confidence_ratio": 1.0,
            },
        ]
    )


class TestRunDailyOrdersUnreadableParams(unittest.TestCase):
    def test_reaches_prediction_sell_and_stops_new_buys(self):
        repo = InMemoryPredictionRepository()
        gate = TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=20_000.0,
        )
        patches = _unreadable_everywhere() + [
            patch.object(repo, "get_latest_by_market", return_value=_predictions()),
            patch("src.trading.execution.runner._record_order"),
            patch("src.trading.execution.sl_tp._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch("src.trading.execution.RiskManager.evaluate_trading_gate", return_value=gate),
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
            patch("src.trading.execution.RiskManager.update_peak_balance"),
            patch(
                "src.trading.execution.runner.evaluate_correlation_gate",
                return_value=CorrelationGateResult(
                    is_allowed=True, enc=2.0, enc_threshold=2.0, avg_correlation=0.0, n_symbols=2
                ),
            ),
            patch("src.trading.execution.runner._load_exit_model", return_value=None),
        ]
        for p in patches:
            p.start()
        broker = _make_broker()
        try:
            stats = run_daily_orders(
                broker,
                order_run_sink=InMemoryOrderRunSink(),
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=repo,
                market="jp",
                mode="paper",
            )
        finally:
            for p in patches:
                p.stop()

        sent = [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]
        self.assertEqual(sent[0], ("7203", OrderSide.SELL))  # SL/TP の決済が先
        self.assertIn(("9002", OrderSide.SELL), sent)  # 予想の売り
        self.assertNotIn(("6501", OrderSide.BUY), sent)  # 新規の買いは止める（#932）
        self.assertGreaterEqual(stats["errors"], 1)


class TestThresholdUnreadable(unittest.TestCase):
    def test_unreadable_uses_default_threshold(self):
        with patch(
            "src.trading.execution.params.get_optimal_params",
            side_effect=OptimalParamsUnreadableError("broken"),
        ):
            with self.assertLogs("src.trading.execution.params", level="ERROR"):
                enriched = _attach_dynamic_thresholds(_predictions())
        self.assertEqual(list(enriched["base_threshold"]), [abs(BUY_THRESHOLD)] * 2)

    def test_readable_keeps_symbol_threshold(self):
        with patch(
            "src.trading.execution.params.get_optimal_params",
            return_value={"threshold": 0.012},
        ):
            enriched = _attach_dynamic_thresholds(_predictions())
        self.assertEqual(list(enriched["base_threshold"]), [0.012, 0.012])


class TestKellyUnreadable(unittest.TestCase):
    def test_unreadable_returns_default_inputs(self):
        with patch(
            "src.trading.execution.params.get_optimal_params",
            side_effect=OptimalParamsUnreadableError("broken"),
        ):
            with self.assertLogs("src.trading.execution.params", level="ERROR"):
                result = _resolve_kelly_params("jp", "6501")
        self.assertEqual(result, (None, None, None))

    def test_readable_keeps_kelly_inputs(self):
        with patch(
            "src.trading.execution.params.get_optimal_params",
            return_value={"metrics": {"win_rate": 0.55, "avg_win": 0.03, "avg_loss": 0.02}},
        ):
            result = _resolve_kelly_params("jp", "6501")
        self.assertEqual(result, (0.55, 0.03, 0.02))

    def test_claude_buy_sizes_with_default_kelly(self):
        from src.trading.claude_agent import _handle_place_order

        broker = _make_broker()
        broker.get_positions.return_value = []
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = TradingGateStatus(
            is_allowed=True, stop_active=False
        )
        risk.calc_position_size.return_value = 100
        stats = {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        with patch(
            "src.trading.execution.params.get_optimal_params",
            side_effect=OptimalParamsUnreadableError("broken"),
        ):
            with patch("src.trading.claude_agent._record_order", create=True):
                _handle_place_order(
                    "6501",
                    "buy",
                    "test",
                    "jp",
                    broker,
                    risk,
                    _predictions(),
                    "paper",
                    stats,
                    trade_diff_sink=InMemoryTradeDiffSink(),
                )

        kwargs = risk.calc_position_size.call_args.kwargs
        self.assertEqual(
            (kwargs["win_rate"], kwargs["avg_win"], kwargs["avg_loss"]), (None, None, None)
        )


if __name__ == "__main__":
    unittest.main()
