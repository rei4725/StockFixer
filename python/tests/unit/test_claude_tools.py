"""
ユニットテスト: trading/claude_tools.py の TOOLS と build_system_prompt、
claude_agent.py の tool の名前の振り分け（#1939）

本物の API は呼ばない。anthropic.Anthropic は MagicMock で差し替える。
"""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# anthropic が未インストールの場合、テスト用スタブを sys.modules に注入する
if "anthropic" not in sys.modules:
    sys.modules["anthropic"] = MagicMock()

from src.infrastructure.in_memory import InMemoryPredictionRepository, InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase
from src.trading.claude_tools import TOOLS, build_system_prompt
from src.trading.types import TradingGateStatus

_TOOL_NAMES = {"get_predictions", "get_positions", "get_balance", "get_risk_status", "place_order"}


class TestTools(unittest.TestCase):
    def test_tool_names_match_five(self):
        self.assertEqual({t["name"] for t in TOOLS}, _TOOL_NAMES)
        self.assertEqual(len(TOOLS), 5)

    def test_every_tool_has_input_schema(self):
        for tool in TOOLS:
            with self.subTest(tool=tool["name"]):
                self.assertIn("input_schema", tool)
                self.assertEqual(tool["input_schema"]["type"], "object")

    def test_place_order_schema(self):
        place = next(t for t in TOOLS if t["name"] == "place_order")
        schema = place["input_schema"]
        self.assertEqual(schema["properties"]["side"]["enum"], ["buy", "sell"])
        self.assertEqual(set(schema["required"]), {"symbol", "side", "reasoning"})


class TestBuildSystemPrompt(unittest.TestCase):
    def test_prompt_contains_limit_notice_market_mode(self):
        notice = "\n## 本日の状態\nテストの止めの知らせ。\n"
        with patch("src.trading.claude_tools.MAX_ORDERS_PER_RUN", 7):
            prompt = build_system_prompt(notice, "jp", "paper")
        self.assertIn("1セッションの最大発注数: 7 件", prompt)
        self.assertIn("テストの止めの知らせ。", prompt)
        self.assertIn("market: jp, mode: paper", prompt)

    def test_prompt_uses_settings_default_limit(self):
        prompt = build_system_prompt("", "us", "paper")
        self.assertIn("1セッションの最大発注数: 5 件", prompt)
        self.assertIn("market: us, mode: paper", prompt)


def _tool_use_response(names):
    response = MagicMock()
    response.stop_reason = "tool_use"
    blocks = []
    for i, name in enumerate(names):
        block = MagicMock()
        block.type = "tool_use"
        block.id = f"tool-{i}"
        block.name = name
        block.input = (
            {"symbol": "7203", "side": "sell", "reasoning": "テスト"}
            if name == "place_order"
            else ({"market": "jp"} if name == "get_predictions" else {})
        )
        blocks.append(block)
    response.content = blocks
    return response


def _end_response():
    response = MagicMock()
    response.stop_reason = "end_turn"
    block = MagicMock()
    block.type = "text"
    block.text = "終了"
    response.content = [block]
    return response


class TestToolDispatch(unittest.TestCase):
    def _run(self, names):
        broker = MagicMock(spec=BrokerBase)
        broker.broker_name = "paper"
        broker.get_balance.return_value = 1_000_000.0
        broker.get_positions.return_value = []
        broker.get_token.return_value = "token"
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason=None,
            reason_code=None,
            daily_loss=0.0,
            daily_loss_limit=20_000.0,
            consecutive_losses=0,
            position_count=0,
            max_positions=10,
        )
        client = MagicMock()
        client.messages.create.side_effect = [_tool_use_response(names), _end_response()]
        repo = InMemoryPredictionRepository()

        from src.trading.claude_agent import run_claude_trader

        with patch("src.trading.claude_agent.RiskManager", return_value=risk), patch.object(
            repo, "get_latest_by_market", return_value=pd.DataFrame()
        ), patch("anthropic.Anthropic", return_value=client), patch(
            "src.trading.claude_agent._attach_dynamic_thresholds", return_value=pd.DataFrame()
        ), patch(
            "src.trading.claude_agent.apply_multi_horizon_score_column",
            return_value=pd.DataFrame(),
        ), patch(
            "src.trading.claude_agent._record_order"
        ):
            run_claude_trader(
                broker=broker,
                market="jp",
                mode="paper",
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=repo,
            )
        messages = client.messages.create.call_args_list[-1].kwargs["messages"]
        return [
            item
            for m in messages
            if m["role"] == "user" and isinstance(m["content"], list)
            for item in m["content"]
        ]

    def test_all_five_tools_dispatched_without_unknown(self):
        names = [
            "get_predictions",
            "get_positions",
            "get_balance",
            "get_risk_status",
            "place_order",
        ]
        results = self._run(names)
        self.assertEqual([r["tool_use_id"] for r in results], [f"tool-{i}" for i in range(5)])
        for r in results:
            with self.subTest(tool_use_id=r["tool_use_id"]):
                self.assertNotIn("未知の tool", r["content"])

    def test_unknown_tool_name_returns_unknown_error(self):
        results = self._run(["no_such_tool"])
        self.assertEqual(len(results), 1)
        self.assertEqual(json.loads(results[0]["content"]), {"error": "未知の tool: no_such_tool"})


if __name__ == "__main__":
    unittest.main()
