"""
ユニットテスト: optimal_params.json が読めない日の閾値・Kelly 入力（#1214）

- 8:50 の道は既定の閾値で予想の売りまで出し、新規の買いは止め、errors を数える
- Claude の発注の道は既定の Kelly 入力で株数を決める
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

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


def _order_patches(repo):
    gate = TradingGateStatus(
        is_allowed=True,
        stop_active=False,
        reason_code=None,
        reason=None,
        daily_loss=0.0,
        daily_loss_limit=20_000.0,
    )
    return [
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


class TestParamsUnreadableReasonIsRecorded(unittest.TestCase):
    """params の門で新規を止めた朝は、理由を stats と 8:50 の記録に残す（#2191）。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dest = Path(tmp.name) / "orders" / "2026-11-26.json"
        for target, value in (
            ("get_results_dir", tmp.name),
            ("now_jst", datetime(2026, 11, 26, 8, 50, tzinfo=ZoneInfo("Asia/Tokyo"))),
        ):
            started = patch(f"src.trading.execution.gate_record.{target}", return_value=value)
            started.start()
            self.addCleanup(started.stop)

    def _run(self, params_patches):
        repo = InMemoryPredictionRepository()
        patches = params_patches + _order_patches(repo)
        for p in patches:
            p.start()
        try:
            return run_daily_orders(
                _make_broker(),
                order_run_sink=InMemoryOrderRunSink(),
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=repo,
                market="jp",
                mode="paper",
            )
        finally:
            for p in patches:
                p.stop()

    def test_unreadable_day_writes_reason_to_stats_and_record(self):
        with self.assertLogs("src.trading.execution.params", level="ERROR"):
            stats = self._run(_unreadable_everywhere())

        reason = stats.get("params_unreadable_reason")
        self.assertIsNotNone(reason)
        self.assertIn("optimal_params.json", reason)
        self.assertIn("2件", reason)  # 読めなかった銘柄の数
        record = json.loads(self.dest.read_text(encoding="utf-8"))
        self.assertEqual(record["params_unreadable_reason"], reason)
        self.assertEqual(record["errors"], stats["errors"])

    def test_readable_day_keeps_reason_null(self):
        readable = [
            patch("src.trading.risk_manager.get_optimal_params", return_value={}),
            patch(
                "src.trading.execution.params.get_optimal_params",
                return_value={"threshold": 0.012},
            ),
        ]
        stats = self._run(readable)

        self.assertIsNone(stats["params_unreadable_reason"])
        record = json.loads(self.dest.read_text(encoding="utf-8"))
        self.assertIn("params_unreadable_reason", record)
        self.assertIsNone(record["params_unreadable_reason"])


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

    def test_all_losses_keeps_zero_win_rate_and_sizes_zero(self):
        # 取引があり全部負けた銘柄は勝率 0 を既定の 55% に戻さず、新規を見送る（#1259）
        from src.trading.risk_manager import RiskManager

        with patch(
            "src.trading.execution.params.get_optimal_params",
            return_value={"metrics": {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.012}},
        ):
            win_rate, avg_win, avg_loss = _resolve_kelly_params("jp", "6501")
        self.assertEqual(win_rate, 0.0)
        self.assertEqual(avg_loss, 0.012)

        risk = RiskManager(_make_broker(), market="jp")
        risk.get_current_dd_ratio = MagicMock(return_value=0.0)
        with patch("yfinance.Ticker", side_effect=RuntimeError("unit テストでは ^VIX を読まない")):
            qty = risk.calc_position_size(
                "6501", price=1000.0, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss
            )
        self.assertEqual(qty, 0)

    def test_unusable_metrics_return_none(self):
        # 取引なし（全部 0）・NaN・欠けは今までどおり既定に戻す
        cases = [
            {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0},
            {"win_rate": float("nan"), "avg_win": float("nan"), "avg_loss": float("nan")},
            {"win_rate": float("nan"), "avg_win": 0.0, "avg_loss": 0.012},
            {"win_rate": 0.0, "avg_win": 0.0},
            {},
        ]
        for metrics in cases:
            with self.subTest(metrics=metrics):
                with patch(
                    "src.trading.execution.params.get_optimal_params",
                    return_value={"metrics": metrics},
                ):
                    win_rate, avg_win, _ = _resolve_kelly_params("jp", "6501")
                self.assertIsNone(win_rate)
                self.assertIsNone(avg_win)

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
