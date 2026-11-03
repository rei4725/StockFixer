"""Claude の道は、今日（JST）の production の予想の保存が終わっているときだけ新規の buy を出す（#1490）。

8:50 の道（#1394）と同じく、最新の回が前の日の回か、保存の終わりの印が今日でなければ、
Claude が buy を頼んでも send_order に届かず skipped に数え、保有の sell は今までどおり出す。
"""

import json
import sys
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = MagicMock()

from src.infrastructure.in_memory import InMemoryPredictionRepository, InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution.pending import PendingExits
from src.trading.types import TradingGateStatus

_PREDICTIONS = InMemoryPredictionRepository()

# 8:50 JST に手で流した時刻（コンテナの時計は UTC）
_NOW_UTC = datetime(2026, 10, 29, 23, 50, tzinfo=timezone.utc)
_TODAY_JST = "20261030"
_TODAY_RUN = "20261029_224343"  # 7:43 JST の今日の回
_YESTERDAY_RUN = "20261028_224343"  # 前の日の回


def _row(symbol, diff_ratio, predicted_at):
    return {
        "market": "jp",
        "symbol": symbol,
        "predicted_at": predicted_at,
        "current_price": 1000.0,
        "diff_ratio": diff_ratio,
        "confidence_ratio": 1.0,
        "multi_horizon_score": diff_ratio,
        "effective_buy_threshold": 0.005,
        "effective_sell_threshold": -0.005,
        "threshold_scale": 1.0,
    }


def _response(stop_reason, tool_calls=()):
    response = MagicMock()
    response.stop_reason = stop_reason
    blocks = []
    for i, (name, tool_input) in enumerate(tool_calls):
        block = MagicMock()
        block.type = "tool_use"
        block.id = f"tool-{i}"
        block.name = name
        block.input = tool_input
        blocks.append(block)
    response.content = blocks
    return response


class TestClaudeTraderStalePrediction(unittest.TestCase):
    def _run(self, predicted_at, saved_day):
        """get_predictions と、7200 の buy・9000 の sell を頼ませ、(stats, 送った注文, 候補) を返す。"""
        from src.trading.claude_agent import run_claude_trader

        df = pd.DataFrame([_row("7200", 0.03, predicted_at), _row("9000", -0.03, predicted_at)])
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        broker.get_balance.return_value = 1_000_000.0
        broker.get_positions.return_value = [{"symbol": "9000", "qty": 100, "avg_price": 1000.0}]
        broker.get_token.return_value = "token"
        broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = TradingGateStatus(
            is_allowed=True, stop_active=False
        )
        risk.calc_position_size.return_value = 100
        client = MagicMock()
        client.messages.create.side_effect = [
            _response(
                "tool_use",
                [
                    ("get_predictions", {"market": "jp"}),
                    ("place_order", {"symbol": "7200", "side": "buy", "reasoning": "テスト"}),
                    ("place_order", {"symbol": "9000", "side": "sell", "reasoning": "テスト"}),
                ],
            ),
            _response("end_turn"),
        ]
        with ExitStack() as stack:

            def p(*args, **kwargs):
                return stack.enter_context(patch(*args, **kwargs))

            stack.enter_context(patch.object(_PREDICTIONS, "get_latest_by_market", return_value=df))
            p("anthropic.Anthropic", return_value=client)
            p("src.trading.claude_agent.RiskManager", return_value=risk)
            p("src.trading.execution.freshness._local_now", return_value=_NOW_UTC)
            p("src.trading.claude_agent._attach_dynamic_thresholds", side_effect=lambda d: d)
            p("src.trading.claude_agent.apply_multi_horizon_score_column", side_effect=lambda d: d)
            p("src.trading.claude_agent._record_order")
            p("src.trading.claude_agent._resolve_kelly_params", return_value=(0.55, 0.015, 0.008))
            p(
                "src.trading.claude_agent._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            )
            p("src.trading.claude_agent._load_pending_exits", return_value=PendingExits())
            p("src.trading.correlation_risk._load_recent_returns", return_value=pd.DataFrame())
            p(
                "src.utils.sector_constraints._lookup_symbol_sector",
                side_effect=lambda market, symbol: f"sector-{symbol}",
            )
            stats = run_claude_trader(
                broker=broker,
                market="jp",
                mode="paper",
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
                prediction_saved_day=saved_day,
            )
        sent = [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]
        tool_results = client.messages.create.call_args_list[1].kwargs["messages"][2]["content"]
        tool_output = json.loads(tool_results[0]["content"])
        system_prompt = client.messages.create.call_args_list[0].kwargs["system"]
        return stats, sent, tool_output, system_prompt

    def test_previous_day_run_blocks_buy_and_keeps_sell(self):
        stats, sent, tool_output, system_prompt = self._run(_YESTERDAY_RUN, _TODAY_JST)

        self.assertEqual(sent, [("9000", OrderSide.SELL)])
        self.assertEqual((stats["buy_orders"], stats["sell_orders"], stats["skipped"]), (0, 1, 1))
        self.assertIn(_YESTERDAY_RUN, stats["stale_prediction_reason"])
        self.assertIn(_YESTERDAY_RUN, system_prompt)
        self.assertEqual(tool_output["buy_candidates"], [])
        self.assertIn(_YESTERDAY_RUN, tool_output["buy_blocked_reason"])

    def test_today_run_without_saved_marker_blocks_buy(self):
        # 回は今日だが保存の途中（印が前の日・無い）も新規の buy を出さない
        for saved_day in ("20261029", ""):
            with self.subTest(saved_day=saved_day):
                stats, sent, tool_output, _ = self._run(_TODAY_RUN, saved_day)
                self.assertEqual(sent, [("9000", OrderSide.SELL)])
                self.assertEqual((stats["buy_orders"], stats["skipped"]), (0, 1))
                self.assertEqual(tool_output["buy_candidates"], [])

    def test_today_run_and_marker_places_buy(self):
        stats, sent, tool_output, _ = self._run(_TODAY_RUN, _TODAY_JST)

        self.assertEqual(sent, [("7200", OrderSide.BUY), ("9000", OrderSide.SELL)])
        self.assertEqual((stats["buy_orders"], stats["sell_orders"], stats["skipped"]), (1, 1, 0))
        self.assertNotIn("stale_prediction_reason", stats)
        self.assertEqual([c["symbol"] for c in tool_output["buy_candidates"]], ["7200"])
        self.assertNotIn("buy_blocked_reason", tool_output)


if __name__ == "__main__":
    unittest.main()
