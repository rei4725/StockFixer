"""
ユニットテスト: ClaudeTrader

anthropic SDK・DuckDB・Broker はすべて MagicMock で差し替える。
anthropic が未インストールの環境でも実行できるよう sys.modules にスタブを注入する。
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# anthropic が未インストールの場合、テスト用スタブを sys.modules に注入する
if "anthropic" not in sys.modules:
    _mock_anthropic = MagicMock()
    sys.modules["anthropic"] = _mock_anthropic

from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.types import TradingGateStatus

# ---------------------------------------------------------------------------
# テスト用ヘルパー
# ---------------------------------------------------------------------------


def _make_broker(balance=1_000_000.0, positions=None):
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = balance
    broker.get_positions.return_value = positions or []
    broker.get_token.return_value = "token"
    broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
    return broker


def _make_gate(is_allowed=True, stop_active=False, reason=None, reason_code=None):
    return TradingGateStatus(
        is_allowed=is_allowed,
        stop_active=stop_active,
        reason=reason,
        reason_code=reason_code,
        daily_loss=0.0,
        daily_loss_limit=20_000.0,
        consecutive_losses=0,
        position_count=0,
        max_positions=10,
    )


def _make_predictions_df(n=3):
    rows = [
        {
            "market": "jp",
            "symbol": f"720{i}",
            "predicted_at": "20260516_063100",
            "current_price": 1000.0 + i * 10,
            "diff_ratio": 0.02 + 0.005 * i,
            "diff_ratio_3d": 0.015,
            "diff_ratio_5d": 0.012,
            "diff_ratio_10d": 0.010,
            "confluence_score": 3,
            "confidence_ratio": 0.9,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def _make_anthropic_response(stop_reason="end_turn", tool_calls=None):
    """anthropic.types.Message の最低限モックを返す"""
    response = MagicMock()
    response.stop_reason = stop_reason
    if tool_calls:
        blocks = []
        for tc in tool_calls:
            block = MagicMock()
            block.type = "tool_use"
            block.id = tc["id"]
            block.name = tc["name"]
            block.input = tc["input"]
            blocks.append(block)
        response.content = blocks
    else:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "判断完了。本日の発注を終了します。"
        response.content = [text_block]
    return response


# ---------------------------------------------------------------------------
# ツールハンドラー単体テスト
# ---------------------------------------------------------------------------


class TestHandlePlaceOrder(unittest.TestCase):
    """_handle_place_order の RiskManager ゲート強制とサイジングを検証"""

    def setUp(self):
        self.broker = _make_broker()
        self.risk = MagicMock()
        self.risk.evaluate_trading_gate.return_value = _make_gate(is_allowed=True)
        self.risk.calc_position_size.return_value = 100
        self.predictions = _make_predictions_df()
        # _attach_dynamic_thresholds, apply_multi_horizon_score_column を通した列を追加
        self.predictions["multi_horizon_score"] = self.predictions["diff_ratio"]
        self.predictions["effective_buy_threshold"] = 0.005
        self.predictions["effective_sell_threshold"] = -0.005
        self.predictions["base_threshold"] = 0.005
        self.predictions["threshold_scale"] = 1.0

    def _call(self, symbol, side_str, stats=None):
        from src.trading.claude_agent import _handle_place_order

        if stats is None:
            stats = {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}
        return _handle_place_order(
            symbol=symbol,
            side_str=side_str,
            reasoning="テスト用の理由",
            market="jp",
            broker=self.broker,
            risk=self.risk,
            predictions_cache=self.predictions,
            mode="paper",
            stats=stats,
        )

    @patch("src.trading.claude_agent._record_order")
    @patch("src.trading.claude_agent._get_held_symbols", return_value=set())
    @patch("src.trading.claude_agent._resolve_kelly_params", return_value=(0.55, 0.015, 0.008))
    @patch(
        "src.trading.claude_agent._choose_order_params",
        return_value=(OrderType.MARKET, 0.0, "market", "open"),
    )
    def test_buy_order_placed_successfully(self, _choose, _kelly, _held, _record):
        result = self._call("7200", "buy")
        self.assertEqual(result["status"], "placed")
        self.broker.send_order.assert_called_once()
        call_args = self.broker.send_order.call_args
        self.assertEqual(call_args[0][1], OrderSide.BUY)

    @patch("src.trading.claude_agent._record_order")
    @patch(
        "src.trading.claude_agent._choose_order_params",
        return_value=(OrderType.MARKET, 0.0, "market", "open"),
    )
    def test_sell_order_uses_position_qty(self, _choose, _record):
        self.broker.get_positions.return_value = [
            {"symbol": "7200", "qty": 200, "avg_price": 990.0, "current_price": 1000.0}
        ]
        result = self._call("7200", "sell")
        self.assertEqual(result["status"], "placed")
        call_args = self.broker.send_order.call_args
        self.assertEqual(call_args[0][1], OrderSide.SELL)
        self.assertEqual(call_args[0][2], 200)

    def test_risk_gate_blocked_skips_order(self):
        self.risk.evaluate_trading_gate.return_value = _make_gate(
            is_allowed=False, reason="日次損失上限"
        )
        stats = {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}
        result = self._call("7200", "buy", stats)
        self.assertEqual(result["status"], "skipped")
        self.broker.send_order.assert_not_called()
        self.assertEqual(stats["skipped"], 1)

    def test_unknown_symbol_skips_order(self):
        result = self._call("UNKNOWN", "buy")
        self.assertEqual(result["status"], "skipped")
        self.broker.send_order.assert_not_called()

    @patch("src.trading.claude_agent._get_held_symbols", return_value=set())
    def test_zero_position_size_skips_order(self, _held):
        self.risk.calc_position_size.return_value = 0
        result = self._call("7200", "buy")
        self.assertEqual(result["status"], "skipped")
        self.broker.send_order.assert_not_called()

    @patch("src.trading.claude_agent._get_held_symbols", return_value=set())
    def test_min_change_ratio_skips_order(self, _held):
        # diff_ratio を MIN_CHANGE_RATIO 未満に設定
        self.predictions.loc[self.predictions["symbol"] == "7200", "diff_ratio"] = 0.0001
        result = self._call("7200", "buy")
        self.assertEqual(result["status"], "skipped")
        self.broker.send_order.assert_not_called()

    def test_sell_with_no_position_skips(self):
        self.broker.get_positions.return_value = []
        result = self._call("7200", "sell")
        self.assertEqual(result["status"], "skipped")

    @patch("src.trading.claude_agent._get_held_symbols", return_value={"7200"})
    def test_buy_already_held_skips(self, _held):
        result = self._call("7200", "buy")
        self.assertEqual(result["status"], "skipped")


# ---------------------------------------------------------------------------
# get_predictions ハンドラーテスト
# ---------------------------------------------------------------------------


class TestHandleGetPredictions(unittest.TestCase):
    @patch("src.trading.claude_agent._load_latest_predictions")
    @patch("src.trading.claude_agent._get_held_symbols", return_value={"7200"})
    def test_returns_buy_and_sell_candidates(self, _held, _load):
        df = _make_predictions_df(n=4)
        # _attach_dynamic_thresholds と apply_multi_horizon_score_column の結果を模倣
        df["multi_horizon_score"] = df["diff_ratio"]
        df["effective_buy_threshold"] = 0.005
        df["effective_sell_threshold"] = -0.005
        df["base_threshold"] = 0.005
        df["threshold_scale"] = 1.0
        _load.return_value = df

        broker = _make_broker()

        with patch("src.trading.claude_agent._attach_dynamic_thresholds", return_value=df), patch(
            "src.trading.claude_agent.apply_multi_horizon_score_column", return_value=df
        ):
            from src.trading.claude_agent import _handle_get_predictions

            result = _handle_get_predictions("jp", broker)

        self.assertIn("buy_candidates", result)
        self.assertIn("sell_candidates", result)
        self.assertIn("held_symbols", result)
        # 7200 は保有中なので buy 候補に含まれない
        buy_symbols = [r["symbol"] for r in result["buy_candidates"]]
        self.assertNotIn("7200", buy_symbols)

    @patch("src.trading.claude_agent._load_latest_predictions")
    def test_empty_predictions_returns_message(self, _load):
        _load.return_value = pd.DataFrame()
        broker = _make_broker()

        from src.trading.claude_agent import _handle_get_predictions

        result = _handle_get_predictions("jp", broker)
        self.assertIn("message", result)


# ---------------------------------------------------------------------------
# run_claude_trader 統合テスト（anthropic モック）
# ---------------------------------------------------------------------------


class TestRunClaudeTrader(unittest.TestCase):
    def _make_risk_mock(self, is_allowed=True):
        risk = MagicMock()
        risk.evaluate_trading_gate.return_value = _make_gate(is_allowed=is_allowed)
        risk.update_peak_balance.return_value = None
        risk.calc_position_size.return_value = 100
        return risk

    @patch("src.trading.claude_agent.RiskManager")
    @patch("src.trading.claude_agent._load_latest_predictions")
    @patch("anthropic.Anthropic")
    def test_returns_stats_on_success(self, mock_anthropic_cls, mock_load, mock_risk_cls):
        broker = _make_broker()
        mock_load.return_value = pd.DataFrame()
        mock_risk_cls.return_value = self._make_risk_mock()

        client = MagicMock()
        client.messages.create.return_value = _make_anthropic_response(stop_reason="end_turn")
        mock_anthropic_cls.return_value = client

        from src.trading.claude_agent import run_claude_trader

        with patch(
            "src.trading.claude_agent._attach_dynamic_thresholds", return_value=pd.DataFrame()
        ), patch(
            "src.trading.claude_agent.apply_multi_horizon_score_column", return_value=pd.DataFrame()
        ):
            stats = run_claude_trader(broker=broker, market="jp", mode="paper")

        self.assertIn("buy_orders", stats)
        self.assertIn("sell_orders", stats)
        self.assertIn("skipped", stats)
        self.assertIn("errors", stats)

    @patch("src.trading.claude_agent.RiskManager")
    @patch("src.trading.claude_agent._load_latest_predictions")
    def test_risk_gate_blocked_returns_early(self, mock_load, mock_risk_cls):
        broker = _make_broker()
        mock_load.return_value = pd.DataFrame()
        risk_mock = self._make_risk_mock(is_allowed=False)
        # stop_active=True かつ reason をセットして正しく stats に反映されるようにする
        risk_mock.evaluate_trading_gate.return_value = _make_gate(
            is_allowed=False, stop_active=True, reason="日次損失上限テスト"
        )
        mock_risk_cls.return_value = risk_mock

        from src.trading.claude_agent import run_claude_trader

        stats = run_claude_trader(broker=broker, market="jp", mode="paper")

        self.assertTrue(stats["trading_stopped"])
        self.assertIsNotNone(stats["stop_reason"])
        # anthropic クライアントが生成されていないこと（buy/sell は 0）
        self.assertEqual(stats["buy_orders"], 0)
        self.assertEqual(stats["sell_orders"], 0)

    @patch("src.trading.claude_agent.RiskManager")
    @patch("src.trading.claude_agent._load_latest_predictions")
    @patch("anthropic.Anthropic")
    def test_tool_use_dispatched_correctly(self, mock_anthropic_cls, mock_load, mock_risk_cls):
        broker = _make_broker()
        mock_load.return_value = pd.DataFrame()
        mock_risk_cls.return_value = self._make_risk_mock()

        # 1回目: get_risk_status tool_use、2回目: end_turn
        tool_call_response = _make_anthropic_response(
            stop_reason="tool_use",
            tool_calls=[
                {
                    "id": "tool-001",
                    "name": "get_risk_status",
                    "input": {},
                }
            ],
        )
        end_response = _make_anthropic_response(stop_reason="end_turn")

        client = MagicMock()
        client.messages.create.side_effect = [tool_call_response, end_response]
        mock_anthropic_cls.return_value = client

        from src.trading.claude_agent import run_claude_trader

        with patch(
            "src.trading.claude_agent._attach_dynamic_thresholds", return_value=pd.DataFrame()
        ), patch(
            "src.trading.claude_agent.apply_multi_horizon_score_column", return_value=pd.DataFrame()
        ):
            run_claude_trader(broker=broker, market="jp", mode="paper")

        # 2回呼ばれているはず（tool_use → end_turn）
        self.assertEqual(client.messages.create.call_count, 2)

    def test_import_error_raised_without_anthropic(self):
        broker = _make_broker()
        with patch.dict("sys.modules", {"anthropic": None}):
            from src.trading.claude_agent import run_claude_trader

            with self.assertRaises((ImportError, TypeError)):
                run_claude_trader(broker=broker)


class TestSaveReasoningLog(unittest.TestCase):
    """save_reasoning_log の ON CONFLICT (run_id) DO UPDATE 経路を実DBで検証する。

    _isolate_db (tests/unit/conftest.py, autouse) が実 Postgres 接続を注入するため、
    ここではモックを使わずDB書き込みを直接確認する。
    """

    def test_insert_then_query_reflects_saved_values(self):
        from src.trading.claude_reasoning_log import save_reasoning_log
        from src.utils.db._connection import _db_connection

        save_reasoning_log(
            run_id="run-001",
            market="jp",
            thinking_text="最初の思考ログ",
            summary="最初の要約",
        )

        with _db_connection() as con:
            row = con.execute(
                "SELECT run_id, market, thinking, summary FROM claude_reasoning WHERE run_id = %s",
                ["run-001"],
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "run-001")
        self.assertEqual(row[1], "jp")
        self.assertEqual(row[2], "最初の思考ログ")
        self.assertEqual(row[3], "最初の要約")

    def test_second_call_with_same_run_id_updates_in_place(self):
        from src.trading.claude_reasoning_log import save_reasoning_log
        from src.utils.db._connection import _db_connection

        save_reasoning_log(
            run_id="run-002",
            market="jp",
            thinking_text="1回目の思考ログ",
            summary="1回目の要約",
        )
        save_reasoning_log(
            run_id="run-002",
            market="us",
            thinking_text="2回目の思考ログ",
            summary="2回目の要約",
        )

        with _db_connection() as con:
            rows = con.execute(
                "SELECT market, thinking, summary FROM claude_reasoning WHERE run_id = %s",
                ["run-002"],
            ).fetchall()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "us")
        self.assertEqual(rows[0][1], "2回目の思考ログ")
        self.assertEqual(rows[0][2], "2回目の要約")


if __name__ == "__main__":
    unittest.main()
