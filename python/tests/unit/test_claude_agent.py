"""
ユニットテスト: ClaudeTrader

anthropic SDK・DuckDB・Broker はすべて MagicMock で差し替える。
anthropic が未インストールの環境でも実行できるよう sys.modules にスタブを注入する。
"""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# anthropic が未インストールの場合、テスト用スタブを sys.modules に注入する
if "anthropic" not in sys.modules:
    _mock_anthropic = MagicMock()
    sys.modules["anthropic"] = _mock_anthropic

from src.infrastructure.in_memory import InMemoryPredictionRepository, InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.types import TradingGateStatus

# claude_agent に注入する予測リポジトリ。テストは get_latest_by_market を patch.object で差し替える
_PREDICTIONS = InMemoryPredictionRepository()

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
        # broker は broker_name="paper" の MagicMock なので、差し替えないとテスト用の DB の
        # paper_orders を読み、ほかのテストの残りで揺れる。約定待ちの決済は既定で空にする
        from src.trading.execution.pending import PendingExits

        pending_patcher = patch(
            "src.trading.claude_agent._load_pending_exits", return_value=PendingExits()
        )
        self.load_pending_exits = pending_patcher.start()
        self.addCleanup(pending_patcher.stop)

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
            trade_diff_sink=InMemoryTradeDiffSink(),
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

    @patch("src.trading.claude_agent._record_order")
    @patch(
        "src.trading.claude_agent._choose_order_params",
        return_value=(OrderType.MARKET, 0.0, "market", "open"),
    )
    @patch("src.trading.execution.pending.db_connection")
    def test_sell_does_not_read_paper_orders(self, mock_db, _choose, _record):
        # #1205: sell の約定待ちの確認は setUp の差し替えを通り、DB に繋がない
        self.broker.get_positions.return_value = [
            {"symbol": "7200", "qty": 200, "avg_price": 990.0, "current_price": 1000.0}
        ]
        result = self._call("7200", "sell")
        self.assertEqual(result["status"], "placed")
        self.load_pending_exits.assert_called_once_with(self.broker)
        mock_db.assert_not_called()

    def test_risk_gate_blocked_skips_order(self):
        self.risk.evaluate_trading_gate.return_value = _make_gate(
            is_allowed=False, reason="日次損失上限"
        )
        stats = {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}
        result = self._call("7200", "buy", stats)
        self.assertEqual(result["status"], "skipped")
        self.broker.send_order.assert_not_called()
        self.assertEqual(stats["skipped"], 1)

    @patch("src.trading.claude_agent._record_order")
    @patch("src.trading.claude_agent._get_held_symbols", return_value=set())
    @patch(
        "src.trading.claude_agent._choose_order_params",
        return_value=(OrderType.MARKET, 0.0, "market", "open"),
    )
    def test_risk_gate_blocked_allows_sell_but_skips_buy(self, _choose, _held, _record):
        # 8:50 の道・ルールの道と同じく、ゲート不合格の日も保有の売りは通す
        self.risk.evaluate_trading_gate.return_value = _make_gate(
            is_allowed=False, stop_active=True, reason="日次損失上限"
        )
        self.broker.get_positions.return_value = [
            {"symbol": "7200", "qty": 200, "avg_price": 990.0, "current_price": 1000.0}
        ]
        stats = {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        sell = self._call("7200", "sell", stats)
        buy = self._call("7201", "buy", stats)

        self.assertEqual(sell["status"], "placed")
        self.assertEqual(buy["status"], "skipped")
        self.assertIn("日次損失上限", buy["reason"])
        self.broker.send_order.assert_called_once()
        self.assertEqual(self.broker.send_order.call_args[0][1], OrderSide.SELL)
        self.assertEqual(stats["sell_orders"], 1)
        self.assertEqual(stats["buy_orders"], 0)
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

    def _pending_from_sent_sells(self, *_args):
        # 紙の SELL は約定まで保有を減らさず paper_orders の約定待ちに入る。送った SELL を約定待ちとして返す
        from src.trading.execution.pending import PendingExits

        sells = frozenset(
            c[0][0] for c in self.broker.send_order.call_args_list if c[0][1] == OrderSide.SELL
        )
        return PendingExits(sells=sells)

    @patch("src.trading.claude_agent._record_order")
    @patch(
        "src.trading.claude_agent._choose_order_params",
        return_value=(OrderType.MARKET, 0.0, "market", "open"),
    )
    def test_second_sell_of_same_symbol_in_run_is_skipped(self, _choose, _record):
        # #1170: 同じ回に同じ銘柄の sell を2回呼んでも SELL は1本だけ
        self.broker.get_positions.return_value = [
            {"symbol": "7200", "qty": 200, "avg_price": 990.0, "current_price": 1000.0}
        ]
        stats = {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}
        with patch(
            "src.trading.claude_agent._load_pending_exits",
            side_effect=self._pending_from_sent_sells,
            create=True,
        ):
            first = self._call("7200", "sell", stats)
            second = self._call("7200", "sell", stats)

        self.assertEqual(first["status"], "placed")
        self.assertEqual(second["status"], "skipped")
        self.assertIn("約定待ち", second["reason"])
        self.broker.send_order.assert_called_once()
        self.assertEqual(stats["sell_orders"], 1)
        self.assertEqual(stats["skipped"], 1)

    def test_sell_skipped_when_previous_run_sell_is_pending(self):
        # #1170: 前の回（16:30 のルールの道など）の SELL が約定待ちなら2本目を出さない
        from src.trading.execution.pending import PendingExits

        self.broker.get_positions.return_value = [
            {"symbol": "7200.T", "qty": 200, "avg_price": 990.0, "current_price": 1000.0}
        ]
        stats = {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}
        with patch(
            "src.trading.claude_agent._load_pending_exits",
            return_value=PendingExits(sells=frozenset({"7200"})),
            create=True,
        ):
            result = self._call("7200", "sell", stats)

        self.assertEqual(result["status"], "skipped")
        self.assertIn("約定待ち", result["reason"])
        self.broker.send_order.assert_not_called()
        self.assertEqual(stats["skipped"], 1)


# ---------------------------------------------------------------------------
# get_predictions ハンドラーテスト
# ---------------------------------------------------------------------------


class TestHandleGetPredictions(unittest.TestCase):
    @patch.object(_PREDICTIONS, "get_latest_by_market")
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

            result = _handle_get_predictions("jp", broker, _PREDICTIONS)

        self.assertIn("buy_candidates", result)
        self.assertIn("sell_candidates", result)
        self.assertIn("held_symbols", result)
        # 7200 は保有中なので buy 候補に含まれない
        buy_symbols = [r["symbol"] for r in result["buy_candidates"]]
        self.assertNotIn("7200", buy_symbols)

    @patch.object(_PREDICTIONS, "get_latest_by_market")
    def test_empty_predictions_returns_message(self, _load):
        _load.return_value = pd.DataFrame()
        broker = _make_broker()

        from src.trading.claude_agent import _handle_get_predictions

        result = _handle_get_predictions("jp", broker, _PREDICTIONS)
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
    @patch.object(_PREDICTIONS, "get_latest_by_market")
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
            stats = run_claude_trader(
                broker=broker,
                market="jp",
                mode="paper",
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
            )

        self.assertIn("buy_orders", stats)
        self.assertIn("sell_orders", stats)
        self.assertIn("skipped", stats)
        self.assertIn("errors", stats)

    @patch("src.trading.claude_agent.RiskManager")
    @patch.object(_PREDICTIONS, "get_latest_by_market")
    @patch("anthropic.Anthropic")
    def test_risk_gate_blocked_keeps_stop_stats(self, mock_anthropic_cls, mock_load, mock_risk_cls):
        broker = _make_broker()
        mock_load.return_value = pd.DataFrame()
        risk_mock = self._make_risk_mock(is_allowed=False)
        # stop_active=True かつ reason をセットして正しく stats に反映されるようにする
        risk_mock.evaluate_trading_gate.return_value = _make_gate(
            is_allowed=False, stop_active=True, reason="日次損失上限テスト"
        )
        mock_risk_cls.return_value = risk_mock
        client = MagicMock()
        client.messages.create.return_value = _make_anthropic_response(stop_reason="end_turn")
        mock_anthropic_cls.return_value = client

        from src.trading.claude_agent import run_claude_trader

        stats = run_claude_trader(
            broker=broker,
            market="jp",
            mode="paper",
            trade_diff_sink=InMemoryTradeDiffSink(),
            prediction_repo=_PREDICTIONS,
        )

        self.assertTrue(stats["trading_stopped"])
        self.assertEqual(stats["stop_reason"], "日次損失上限テスト")
        self.assertEqual(stats["buy_orders"], 0)
        self.assertEqual(stats["sell_orders"], 0)
        # 早く戻らず Claude を呼び、新規の買いを止めることをプロンプトで伝える
        client.messages.create.assert_called_once()
        system_prompt = client.messages.create.call_args.kwargs["system"]
        self.assertIn("日次損失上限テスト", system_prompt)

    @patch("src.trading.claude_agent.RiskManager")
    @patch.object(_PREDICTIONS, "get_latest_by_market")
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
            run_claude_trader(
                broker=broker,
                market="jp",
                mode="paper",
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
            )

        # 2回呼ばれているはず（tool_use → end_turn）
        self.assertEqual(client.messages.create.call_count, 2)

    def _run_with_buys(
        self,
        symbols,
        positions=None,
        orders=None,
        returns=None,
        sectors=None,
        gate=None,
        pending=None,
        shorts=None,
    ):
        """1回の中で symbols の順に place_order の買いを呼ばせ、(stats, broker, 結果) を返す

        orders を渡すと (銘柄, side) の順に呼ばせる。returns は相関に使う終値のリターン、
        sectors は銘柄から業種への対応（無い銘柄は銘柄ごとに別の業種）。
        gate を渡すと、リスクのゲートの評価をその値にする。
        pending を渡すと、broker の get_pending_buy_symbols がそれを返す（例外なら投げる）。
        渡さないときは broker に get_pending_buy_symbols が無い。
        shorts を渡すと、broker の get_short_positions がそれを返す。
        渡さないときは broker に get_short_positions が無い。
        """
        from src.trading.claude_agent import run_claude_trader

        if orders is None:
            orders = [(s, "buy") for s in symbols]
        broker = _make_broker(positions=positions)
        if isinstance(pending, Exception):
            broker.get_pending_buy_symbols = MagicMock(side_effect=pending)
        elif pending is not None:
            broker.get_pending_buy_symbols = MagicMock(return_value=set(pending))
        if shorts is not None:
            broker.get_short_positions = MagicMock(return_value=shorts)
        held = [p["symbol"] for p in positions or []]
        n = max(len(orders), 1)
        df = _make_predictions_df(n=n + len(held))
        df["symbol"] = [f"S{i}" for i in range(n)] + held
        df["multi_horizon_score"] = df["diff_ratio"]
        df["effective_buy_threshold"] = 0.005
        df["effective_sell_threshold"] = -0.005
        tool_calls = [
            {
                "id": f"tool-{i}",
                "name": "place_order",
                "input": {"symbol": s, "side": side, "reasoning": "テスト"},
            }
            for i, (s, side) in enumerate(orders)
        ]
        sector_map = sectors or {}
        client = MagicMock()
        client.messages.create.side_effect = [
            _make_anthropic_response(stop_reason="tool_use", tool_calls=tool_calls),
            _make_anthropic_response(stop_reason="end_turn"),
        ]
        with patch("src.trading.claude_agent.RiskManager") as mock_risk_cls, patch.object(
            _PREDICTIONS, "get_latest_by_market", return_value=df
        ), patch("anthropic.Anthropic", return_value=client), patch(
            "src.trading.claude_agent._attach_dynamic_thresholds", return_value=df
        ), patch(
            "src.trading.claude_agent.apply_multi_horizon_score_column", return_value=df
        ), patch(
            "src.trading.claude_agent._record_order"
        ), patch(
            "src.trading.claude_agent._resolve_kelly_params", return_value=(0.55, 0.015, 0.008)
        ), patch(
            "src.trading.claude_agent._choose_order_params",
            return_value=(OrderType.MARKET, 0.0, "market", "open"),
        ), patch(
            "src.trading.correlation_risk._load_recent_returns",
            return_value=pd.DataFrame() if returns is None else returns,
        ), patch(
            "src.utils.sector_constraints._lookup_symbol_sector",
            side_effect=lambda market, symbol: sector_map.get(symbol, f"sector-{symbol}"),
        ):
            risk_mock = self._make_risk_mock()
            if gate is not None:
                risk_mock.evaluate_trading_gate.return_value = gate
            mock_risk_cls.return_value = risk_mock
            stats = run_claude_trader(
                broker=broker,
                market="jp",
                mode="paper",
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
            )
        tool_results = client.messages.create.call_args_list[1].kwargs["messages"][-1]["content"]
        results = [json.loads(r["content"]) for r in tool_results]
        return stats, broker, results

    def test_buys_beyond_max_orders_per_run_are_skipped(self):
        from config.settings import MAX_ORDERS_PER_RUN

        symbols = [f"S{i}" for i in range(MAX_ORDERS_PER_RUN + 2)]
        stats, broker, results = self._run_with_buys(symbols)

        self.assertEqual(broker.send_order.call_count, MAX_ORDERS_PER_RUN)
        self.assertEqual(stats["buy_orders"], MAX_ORDERS_PER_RUN)
        statuses = [r["status"] for r in results]
        self.assertEqual(statuses[:MAX_ORDERS_PER_RUN], ["placed"] * MAX_ORDERS_PER_RUN)
        self.assertEqual(statuses[MAX_ORDERS_PER_RUN:], ["skipped", "skipped"])
        self.assertIn("MAX_ORDERS_PER_RUN", results[-1]["reason"])

    def test_second_buy_skipped_when_positions_reach_max(self):
        from config.settings import MAX_POSITIONS

        positions = [
            {"symbol": f"H{i}", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}
            for i in range(MAX_POSITIONS - 1)
        ]
        stats, broker, results = self._run_with_buys(["S0", "S1"], positions=positions)

        self.assertEqual([r["status"] for r in results], ["placed", "skipped"])
        self.assertIn("MAX_POSITIONS", results[1]["reason"])
        broker.send_order.assert_called_once()

    def test_second_buy_of_same_symbol_in_run_is_skipped(self):
        stats, broker, results = self._run_with_buys(["S0", "S0"])

        self.assertEqual([r["status"] for r in results], ["placed", "skipped"])
        broker.send_order.assert_called_once()
        self.assertEqual(stats["buy_orders"], 1)

    def test_correlation_gate_blocks_buy_but_allows_sell(self):
        positions = [
            {"symbol": h, "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}
            for h in ("H0", "H1")
        ]
        # 保有の2銘柄がまったく同じ値動き → ENC=1 < 閾値で相関ゲート不合格
        base = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005, -0.015, 0.01, 0.0, 0.012]
        returns = pd.DataFrame({"H0": base, "H1": base})
        stats, broker, results = self._run_with_buys(
            [], positions=positions, orders=[("S0", "buy"), ("H0", "sell")], returns=returns
        )

        self.assertEqual([r["status"] for r in results], ["skipped", "placed"])
        self.assertIn("相関", results[0]["reason"])
        broker.send_order.assert_called_once()
        self.assertEqual(broker.send_order.call_args[0][1], OrderSide.SELL)
        self.assertEqual(stats["buy_orders"], 0)
        self.assertEqual(stats["sell_orders"], 1)

    def test_risk_gate_blocked_at_start_still_sells_held_position(self):
        positions = [{"symbol": "H0", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}]
        gate = _make_gate(is_allowed=False, stop_active=True, reason="日次損失上限テスト")
        stats, broker, results = self._run_with_buys(
            [], positions=positions, orders=[("S0", "buy"), ("H0", "sell")], gate=gate
        )

        self.assertEqual([r["status"] for r in results], ["skipped", "placed"])
        broker.send_order.assert_called_once()
        self.assertEqual(broker.send_order.call_args[0][0], "H0")
        self.assertEqual(broker.send_order.call_args[0][1], OrderSide.SELL)
        self.assertEqual(broker.send_order.call_args[0][2], 100)
        self.assertEqual(stats["buy_orders"], 0)
        self.assertEqual(stats["sell_orders"], 1)
        self.assertTrue(stats["trading_stopped"])
        self.assertEqual(stats["stop_reason"], "日次損失上限テスト")

    def test_same_sector_buy_skipped_when_held_and_run_buys_reach_cap(self):
        from config.settings import MAX_SECTOR_POSITIONS

        positions = [{"symbol": "H0", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}]
        n_buys = MAX_SECTOR_POSITIONS  # 保有の1件と合わせると上限を1件超える
        symbols = [f"S{i}" for i in range(n_buys + 1)]
        sectors = {s: "Tech" for s in ["H0"] + symbols[:n_buys]}
        sectors[symbols[-1]] = "Energy"
        stats, broker, results = self._run_with_buys(symbols, positions=positions, sectors=sectors)

        statuses = [r["status"] for r in results]
        self.assertEqual(statuses, ["placed"] * (n_buys - 1) + ["skipped", "placed"])
        self.assertIn("MAX_SECTOR_POSITIONS", results[n_buys - 1]["reason"])
        self.assertEqual(stats["buy_orders"], n_buys)

    def _nine_held(self):
        return [
            {"symbol": f"H{i}", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}
            for i in range(9)
        ]

    def test_pending_buy_counts_toward_max_positions(self):
        # 約定済み9件＋前の回の約定待ちの BUY 1件で MAX_POSITIONS=10 に達する（#1177）
        with patch("src.trading.claude_order_caps.MAX_POSITIONS", 10):
            stats, broker, results = self._run_with_buys(
                ["S0"], positions=self._nine_held(), pending={"P0"}
            )

        self.assertEqual(results[0]["status"], "skipped")
        self.assertIn("MAX_POSITIONS", results[0]["reason"])
        broker.send_order.assert_not_called()
        self.assertEqual(stats["buy_orders"], 0)

    def test_pending_buy_counts_toward_sector_cap(self):
        with patch("src.trading.claude_order_caps.MAX_SECTOR_POSITIONS", 1):
            stats, broker, results = self._run_with_buys(
                ["S0", "S1"], pending={"P0"}, sectors={"P0": "Tech", "S0": "Tech"}
            )

        self.assertEqual([r["status"] for r in results], ["skipped", "placed"])
        self.assertIn("MAX_SECTOR_POSITIONS", results[0]["reason"])
        self.assertEqual(broker.send_order.call_args[0][0], "S1")

    def test_pending_buys_enter_correlation_gate(self):
        base = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005, -0.015, 0.01, 0.0, 0.012]
        returns = pd.DataFrame({"P0": base, "P1": base})
        stats, broker, results = self._run_with_buys(["S0"], pending={"P0", "P1"}, returns=returns)

        self.assertEqual(results[0]["status"], "skipped")
        self.assertIn("相関", results[0]["reason"])
        broker.send_order.assert_not_called()

    def test_pending_buy_symbol_is_treated_as_held(self):
        stats, broker, results = self._run_with_buys(["S0"], pending={"S0"})

        self.assertEqual(results[0]["status"], "skipped")
        self.assertIn("既に保有中", results[0]["reason"])
        broker.send_order.assert_not_called()

    def test_broker_without_pending_reader_keeps_current_behavior(self):
        with patch("src.trading.claude_order_caps.MAX_POSITIONS", 10):
            stats, broker, results = self._run_with_buys(["S0"], positions=self._nine_held())

        self.assertEqual(results[0]["status"], "placed")
        broker.send_order.assert_called_once()

    def test_unreadable_pending_buys_keep_current_behavior(self):
        with patch("src.trading.claude_order_caps.MAX_POSITIONS", 10):
            stats, broker, results = self._run_with_buys(
                ["S0"], positions=self._nine_held(), pending=RuntimeError("DB 読めない")
            )

        self.assertEqual(results[0]["status"], "placed")
        broker.send_order.assert_called_once()
        self.assertEqual(stats["errors"], 0)

    def test_short_positions_count_toward_max_positions(self):
        # 買い持ち7件＋空売り2件＋この回の買い1件で MAX_POSITIONS=10 に達する（#1211）
        shorts = [{"symbol": f"T{i}", "qty": 100, "unrealized_pnl": 0.0} for i in range(2)]
        with patch("src.trading.claude_order_caps.MAX_POSITIONS", 10):
            stats, broker, results = self._run_with_buys(
                ["S0", "S1"], positions=self._nine_held()[:7], shorts=shorts
            )

        self.assertEqual([r["status"] for r in results], ["placed", "skipped"])
        self.assertIn("MAX_POSITIONS", results[1]["reason"])
        broker.send_order.assert_called_once()
        self.assertEqual(stats["buy_orders"], 1)

    def test_closed_short_with_zero_qty_is_not_counted(self):
        shorts = [{"symbol": "T0", "qty": 0, "unrealized_pnl": 0.0}]
        with patch("src.trading.claude_order_caps.MAX_POSITIONS", 10):
            stats, broker, results = self._run_with_buys(
                ["S0"], positions=self._nine_held(), shorts=shorts
            )

        self.assertEqual(results[0]["status"], "placed")
        broker.send_order.assert_called_once()

    def test_import_error_raised_without_anthropic(self):
        broker = _make_broker()
        with patch.dict("sys.modules", {"anthropic": None}):
            from src.trading.claude_agent import run_claude_trader

            with self.assertRaises((ImportError, TypeError)):
                run_claude_trader(
                    broker=broker,
                    trade_diff_sink=InMemoryTradeDiffSink(),
                    prediction_repo=_PREDICTIONS,
                )


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
