"""
ユニットテスト: SL/TP 自動損切り・利確

- RiskManager.check_sl_tp の判定ロジック
- _check_sl_tp_exits の発注・stats 更新
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from config.settings import DEFAULT_STOP_LOSS_PCT
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import _check_sl_tp_exits
from src.trading.risk_manager import RiskManager
from src.utils.optimal_params_loader import OptimalParamsUnreadableError


def _make_broker(positions=None, short_positions=None):
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = positions or []
    broker.get_short_positions.return_value = short_positions or []
    broker.send_order.return_value = {"order_id": "test-sl-001", "status": "pending"}
    return broker


def _make_stats():
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


class TestCheckSlTp(unittest.TestCase):
    def _make_risk(self, stop_loss_pct=None, take_profit_pct=None):
        broker = MagicMock(spec=BrokerBase)
        risk = RiskManager(broker)
        risk.stop_loss_pct = stop_loss_pct
        risk.take_profit_pct = take_profit_pct
        return risk

    def test_sl_triggered(self):
        risk = self._make_risk(stop_loss_pct=0.05)
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=940.0)
        self.assertTrue(sl)
        self.assertFalse(tp)
        self.assertIn("SL発動", reason)

    def test_tp_triggered(self):
        risk = self._make_risk(take_profit_pct=0.10)
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=1110.0)
        self.assertFalse(sl)
        self.assertTrue(tp)
        self.assertIn("TP発動", reason)

    def test_neither_triggered(self):
        risk = self._make_risk(stop_loss_pct=0.05, take_profit_pct=0.10)
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=1020.0)
        self.assertFalse(sl)
        self.assertFalse(tp)
        self.assertEqual(reason, "")

    def test_no_thresholds(self):
        risk = self._make_risk()
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=500.0)
        self.assertFalse(sl)
        self.assertFalse(tp)

    def test_avg_price_zero(self):
        risk = self._make_risk(stop_loss_pct=0.05)
        sl, tp, reason = risk.check_sl_tp(avg_price=0.0, current_price=1000.0)
        self.assertFalse(sl)
        self.assertFalse(tp)

    def test_sl_takes_priority_over_tp(self):
        # SL と TP の両方が設定されていても SL が優先される
        risk = self._make_risk(stop_loss_pct=0.01, take_profit_pct=0.01)
        # pnl_pct = -0.05 → SL ≥ 0.01 のみ発動
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=950.0)
        self.assertTrue(sl)
        self.assertFalse(tp)

    def test_exact_sl_boundary(self):
        risk = self._make_risk(stop_loss_pct=0.05)
        # pnl_pct = -0.05 == -stop_loss_pct → 発動
        sl, tp, _ = risk.check_sl_tp(avg_price=1000.0, current_price=950.0)
        self.assertTrue(sl)

    def test_exact_tp_boundary(self):
        risk = self._make_risk(take_profit_pct=0.10)
        # pnl_pct = 0.10 == take_profit_pct → 発動
        sl, tp, _ = risk.check_sl_tp(avg_price=1000.0, current_price=1100.0)
        self.assertTrue(tp)

    def test_short_rise_triggers_sl(self):
        # 空売りは値上がりが損。売り建て 1000 → 1060 で損益率 -6%（#1120）
        risk = self._make_risk(stop_loss_pct=0.05, take_profit_pct=0.10)
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=1060.0, is_short=True)
        self.assertTrue(sl)
        self.assertFalse(tp)
        self.assertIn("SL発動", reason)

    def test_short_fall_triggers_tp_not_sl(self):
        # 空売りは値下がりが益。売り建て 1000 → 880 で損益率 +12%
        risk = self._make_risk(stop_loss_pct=0.05, take_profit_pct=0.10)
        sl, tp, reason = risk.check_sl_tp(avg_price=1000.0, current_price=880.0, is_short=True)
        self.assertFalse(sl)
        self.assertTrue(tp)
        self.assertIn("TP発動", reason)


class TestCheckSlTpShortExits(unittest.TestCase):
    """空売りの保有も SL/TP で SHORT_COVER の成行で返済する（#1120）"""

    _PARAMS = {"stop_loss_pct": 0.05, "take_profit_pct": 0.10}

    def _run(self, broker, stats):
        with patch("src.trading.risk_manager.get_optimal_params", return_value=self._PARAMS):
            with patch("src.trading.execution.sl_tp._record_order") as record:
                triggered = _check_sl_tp_exits(
                    broker, "jp", "paper", MagicMock(), stats, InMemoryTradeDiffSink()
                )
        return triggered, record

    def test_short_rise_past_sl_sends_short_cover(self):
        shorts = [
            {"symbol": "6758", "qty": 100, "avg_short_price": 1000.0, "current_price": 1060.0}
        ]
        broker = _make_broker(short_positions=shorts)
        stats = _make_stats()

        triggered, record = self._run(broker, stats)

        broker.send_order.assert_called_once_with(
            "6758", OrderSide.SHORT_COVER, 100, price=0.0, order_type=OrderType.MARKET
        )
        self.assertEqual(record.call_args.kwargs["side"], OrderSide.SHORT_COVER)
        self.assertIn("6758", triggered)
        self.assertEqual(stats["short_orders"], 1)
        self.assertEqual(stats["sell_orders"], 0)

    def test_short_fall_past_tp_sends_short_cover(self):
        shorts = [{"symbol": "6758", "qty": 100, "avg_short_price": 1000.0, "current_price": 880.0}]
        broker = _make_broker(short_positions=shorts)
        stats = _make_stats()

        triggered, _ = self._run(broker, stats)

        broker.send_order.assert_called_once_with(
            "6758", OrderSide.SHORT_COVER, 100, price=0.0, order_type=OrderType.MARKET
        )
        self.assertIn("6758", triggered)

    def test_short_small_fall_does_not_hit_sl(self):
        # 値下がり 6% は空売りの益。SL にも TP（10%）にも当たらない
        shorts = [{"symbol": "6758", "qty": 100, "avg_short_price": 1000.0, "current_price": 940.0}]
        broker = _make_broker(short_positions=shorts)
        stats = _make_stats()

        triggered, _ = self._run(broker, stats)

        broker.send_order.assert_not_called()
        self.assertEqual(triggered, set())

    def test_unpriced_short_is_skipped_with_warning(self):
        shorts = [
            {"symbol": "6758", "qty": 100, "avg_short_price": 1000.0, "current_price": float("nan")}
        ]
        broker = _make_broker(short_positions=shorts)
        stats = _make_stats()

        with self.assertLogs("src.trading.execution.sl_tp", level="WARNING") as logs:
            triggered, _ = self._run(broker, stats)

        broker.send_order.assert_not_called()
        self.assertEqual(triggered, set())
        self.assertTrue(any("6758" in m for m in logs.output))

    def test_broker_without_short_positions_keeps_long_only(self):
        # get_short_positions が既定の空リストのブローカーでは、買い持ちだけを見る今の動き
        positions = [{"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 940.0}]
        broker = _make_broker(positions)
        stats = _make_stats()

        triggered, _ = self._run(broker, stats)

        broker.send_order.assert_called_once_with(
            "7203", OrderSide.SELL, 100, price=0.0, order_type=OrderType.MARKET
        )
        self.assertEqual(triggered, {"7203"})
        self.assertEqual(stats["short_orders"], 0)


class TestCheckSlTpExits(unittest.TestCase):
    def _patch_risk(self, sl=False, tp=False, reason=""):
        return patch(
            "src.trading.execution.RiskManager.check_sl_tp",
            return_value=(sl, tp, reason),
        )

    def _make_market_data(self):
        return MagicMock()

    def test_sl_triggers_sell_order(self):
        positions = [{"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 940.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=True, reason="SL発動: pnl=-6.00% <= -5.00%"):
            with patch("src.trading.execution.params.get_optimal_params", return_value={}):
                with patch("src.trading.execution.sl_tp._record_order"):
                    triggered = _check_sl_tp_exits(
                        broker, "jp", "paper", market_data, stats, InMemoryTradeDiffSink()
                    )

        broker.send_order.assert_called_once_with(
            "7203", OrderSide.SELL, 100, price=0.0, order_type=OrderType.MARKET
        )
        self.assertIn("7203", triggered)
        self.assertEqual(stats["sell_orders"], 1)
        self.assertAlmostEqual(stats["total_turnover"], 940.0 * 100)

    def test_tp_triggers_sell_order(self):
        positions = [{"symbol": "9984", "qty": 200, "avg_price": 1000.0, "current_price": 1150.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(tp=True, reason="TP発動: pnl=15.00% >= 10.00%"):
            with patch("src.trading.execution.params.get_optimal_params", return_value={}):
                with patch("src.trading.execution.sl_tp._record_order"):
                    triggered = _check_sl_tp_exits(
                        broker, "jp", "paper", market_data, stats, InMemoryTradeDiffSink()
                    )

        self.assertIn("9984", triggered)
        self.assertEqual(stats["sell_orders"], 1)

    def test_no_trigger_no_order(self):
        positions = [{"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 1010.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=False, tp=False, reason=""):
            with patch("src.trading.execution.params.get_optimal_params", return_value={}):
                triggered = _check_sl_tp_exits(
                    broker, "jp", "paper", market_data, stats, InMemoryTradeDiffSink()
                )

        broker.send_order.assert_not_called()
        self.assertEqual(len(triggered), 0)
        self.assertEqual(stats["sell_orders"], 0)

    def test_empty_positions_returns_empty_set(self):
        broker = _make_broker(positions=[])
        stats = _make_stats()
        market_data = self._make_market_data()

        triggered = _check_sl_tp_exits(
            broker, "jp", "paper", market_data, stats, InMemoryTradeDiffSink()
        )
        self.assertEqual(triggered, set())
        broker.send_order.assert_not_called()

    def test_zero_qty_skipped(self):
        positions = [{"symbol": "7203", "qty": 0, "avg_price": 1000.0, "current_price": 900.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=True, reason="SL発動"):
            triggered = _check_sl_tp_exits(
                broker, "jp", "paper", market_data, stats, InMemoryTradeDiffSink()
            )

        broker.send_order.assert_not_called()
        self.assertEqual(len(triggered), 0)

    def test_order_error_increments_errors(self):
        positions = [{"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 900.0}]
        broker = _make_broker(positions)
        broker.send_order.side_effect = RuntimeError("send_order failed")
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=True, reason="SL発動"):
            with patch("src.trading.execution.params.get_optimal_params", return_value={}):
                triggered = _check_sl_tp_exits(
                    broker, "jp", "paper", market_data, stats, InMemoryTradeDiffSink()
                )

        self.assertEqual(stats["errors"], 1)
        self.assertEqual(len(triggered), 0)

    def test_unpriced_position_is_not_treated_as_zero_pnl(self):
        """現在値が NaN の保有は損益0として素通りさせず、評価できないと WARNING を出す（#844）"""
        positions = [
            {"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": float("nan")}
        ]
        broker = _make_broker(positions)
        stats = _make_stats()

        with patch("src.trading.execution.RiskManager.check_sl_tp") as check:
            with self.assertLogs("src.trading.execution.sl_tp", level="WARNING") as logs:
                triggered = _check_sl_tp_exits(
                    broker, "jp", "paper", MagicMock(), stats, InMemoryTradeDiffSink()
                )

        check.assert_not_called()
        broker.send_order.assert_not_called()
        self.assertEqual(triggered, set())
        self.assertTrue(any("7203" in m for m in logs.output))

    def test_unpriced_long_and_short_are_counted_in_stats(self):
        """評価できなかった保有の数と銘柄を stats に残し、errors は増やさない（#1296）"""
        positions = [
            {"symbol": "7203.T", "qty": 100, "avg_price": 1000.0, "current_price": float("nan")}
        ]
        shorts = [
            {"symbol": "6758", "qty": 100, "avg_short_price": 1000.0, "current_price": float("nan")}
        ]
        broker = _make_broker(positions, short_positions=shorts)
        stats = _make_stats()

        triggered = _check_sl_tp_exits(
            broker, "jp", "paper", MagicMock(), stats, InMemoryTradeDiffSink()
        )

        broker.send_order.assert_not_called()
        self.assertEqual(triggered, set())
        self.assertEqual(len(stats["sl_tp_unevaluable_symbols"]), 2)
        self.assertEqual(stats["sl_tp_unevaluable_symbols"], ["7203", "6758"])
        self.assertEqual(stats["errors"], 0)

    def test_symbol_dot_t_stripped(self):
        positions = [{"symbol": "7203.T", "qty": 100, "avg_price": 1000.0, "current_price": 900.0}]
        broker = _make_broker(positions)
        stats = _make_stats()
        market_data = self._make_market_data()

        with self._patch_risk(sl=True, reason="SL発動"):
            with patch("src.trading.execution.params.get_optimal_params", return_value={}):
                with patch("src.trading.execution.sl_tp._record_order"):
                    triggered = _check_sl_tp_exits(
                        broker, "jp", "paper", market_data, stats, InMemoryTradeDiffSink()
                    )

        # send_order には .T 除去後の symbol が渡されること
        call_args = broker.send_order.call_args
        self.assertEqual(call_args[0][0], "7203")
        self.assertIn("7203", triggered)

    def test_sl_on_illiquid_wide_range_symbol_sends_market_order(self):
        # #881: 新規の発注なら指値になる銘柄（低出来高・広い値幅）でも、SL/TP の決済は成行で出す
        positions = [{"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 940.0}]
        broker = _make_broker(positions)
        stats = _make_stats()

        with self._patch_risk(sl=True, reason="SL発動: pnl=-6.00% <= -5.00%"):
            with patch(
                "src.trading.execution.params._load_execution_metrics",
                return_value=(10_000.0, 0.03),
            ):
                with patch("src.trading.execution.sl_tp._record_order") as record:
                    with self.assertLogs("src.trading.execution.sl_tp", level="INFO") as logs:
                        triggered = _check_sl_tp_exits(
                            broker, "jp", "paper", MagicMock(), stats, InMemoryTradeDiffSink()
                        )

        broker.send_order.assert_called_once_with(
            "7203", OrderSide.SELL, 100, price=0.0, order_type=OrderType.MARKET
        )
        self.assertEqual(record.call_args.kwargs["order_type"], OrderType.MARKET)
        self.assertEqual(record.call_args.kwargs["order_price"], 0.0)
        self.assertTrue(any("成行売り" in line for line in logs.output))
        self.assertIn("7203", triggered)

    def test_new_order_on_illiquid_wide_range_symbol_stays_limit(self):
        # ふつうの新規の発注の指値の判定は変わらない
        from src.trading.execution.params import _choose_order_params

        with patch(
            "src.trading.execution.params._load_execution_metrics",
            return_value=(10_000.0, 0.03),
        ):
            order_type, price, _, session = _choose_order_params(
                "jp", "7203", OrderSide.SELL, 1000.0, MagicMock()
            )

        self.assertEqual(order_type, OrderType.LIMIT)
        self.assertGreater(price, 0.0)
        self.assertEqual(session, "close")


class TestSlTpUnreadableParams(unittest.TestCase):
    """optimal_params.json が読めない日も、既定の損切りの線で SL を判定する（#1192）"""

    _LOSS = DEFAULT_STOP_LOSS_PCT + 0.02  # 既定の線を超えた損益率
    _POSITIONS = [
        {"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0 * (1 - _LOSS)},
        {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
        {"symbol": "9002", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0},
    ]
    _SHORTS = [
        {
            "symbol": "6758",
            "qty": 100,
            "avg_short_price": 1000.0,
            "current_price": 1000.0 * (1 + _LOSS),
        }
    ]

    @staticmethod
    def _unreadable():
        return patch(
            "src.trading.risk_manager.get_optimal_params",
            side_effect=OptimalParamsUnreadableError("broken"),
        )

    @staticmethod
    def _sent(broker):
        return [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]

    def test_default_stop_exits_long_and_short(self):
        broker = _make_broker(self._POSITIONS, self._SHORTS)
        stats = _make_stats()

        with self._unreadable():
            with patch("src.trading.execution.sl_tp._record_order"):
                with self.assertLogs("src.trading.execution.sl_tp", level="ERROR"):
                    triggered = _check_sl_tp_exits(
                        broker, "jp", "paper", MagicMock(), stats, InMemoryTradeDiffSink()
                    )

        self.assertEqual(
            sorted(self._sent(broker)),
            sorted([("7203", OrderSide.SELL), ("6758", OrderSide.SHORT_COVER)]),
        )
        self.assertEqual(triggered, {"7203", "6758"})
        self.assertGreaterEqual(stats["errors"], 1)

    def test_exits_only_path_uses_default_stop(self):
        # 予測を使えない日の SL/TP だけの道も同じ扱い（#932・#1049）
        from src.trading.execution.sl_tp import _sl_tp_exits_only
        from src.trading.types import TradingGateStatus

        gate = TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=20_000.0,
        )
        broker = _make_broker(self._POSITIONS, self._SHORTS)
        stats = _make_stats()

        with self._unreadable():
            with patch("src.trading.execution.sl_tp._record_order"):
                _sl_tp_exits_only(
                    broker, "jp", "paper", MagicMock(), stats, InMemoryTradeDiffSink(), gate
                )

        self.assertEqual(
            sorted(self._sent(broker)),
            sorted([("7203", OrderSide.SELL), ("6758", OrderSide.SHORT_COVER)]),
        )
        self.assertGreaterEqual(stats["errors"], 1)

    def test_run_daily_orders_reaches_prediction_sell(self):
        from src.infrastructure.in_memory import (
            InMemoryOrderRunSink,
            InMemoryPredictionRepository,
        )
        from src.trading.execution import run_daily_orders
        from src.trading.types import CorrelationGateResult, TradingGateStatus

        repo = InMemoryPredictionRepository()
        predictions = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "9002",
                    "predicted_at": "20261026_083000",
                    "current_price": 1000.0,
                    "diff_ratio": -0.03,
                    "confidence_ratio": 1.0,
                }
            ]
        )
        gate = TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            reason_code=None,
            reason=None,
            daily_loss=0.0,
            daily_loss_limit=20_000.0,
        )
        patches = [
            self._unreadable(),
            patch.object(repo, "get_latest_by_market", return_value=predictions),
            patch("src.trading.execution.runner._record_order"),
            patch("src.trading.execution.sl_tp._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=gate,
            ),
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
            patch("src.trading.execution.RiskManager.update_peak_balance"),
            patch(
                "src.trading.execution.runner.evaluate_correlation_gate",
                return_value=CorrelationGateResult(
                    is_allowed=True, enc=2.0, enc_threshold=2.0, avg_correlation=0.0, n_symbols=3
                ),
            ),
            patch("src.trading.execution.runner._load_exit_model", return_value=None),
            patch("src.trading.execution.params.get_optimal_params", return_value={}),
        ]
        for p in patches:
            p.start()
        broker = _make_broker(self._POSITIONS, self._SHORTS)
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

        sent = self._sent(broker)
        self.assertIn(("7203", OrderSide.SELL), sent)
        self.assertIn(("6758", OrderSide.SHORT_COVER), sent)
        self.assertIn(("9002", OrderSide.SELL), sent)
        self.assertNotIn(("9000", OrderSide.SELL), sent)
        self.assertGreaterEqual(stats["errors"], 1)

    def test_readable_params_keep_symbol_thresholds(self):
        # 読める日は銘柄ごとの線（5%）で判定し、既定の線に置き換えない
        positions = [{"symbol": "7203", "qty": 100, "avg_price": 1000.0, "current_price": 940.0}]
        broker = _make_broker(positions)
        stats = _make_stats()

        with patch(
            "src.trading.risk_manager.get_optimal_params",
            return_value={"stop_loss_pct": 0.05, "take_profit_pct": 0.10},
        ):
            with patch("src.trading.execution.sl_tp._record_order"):
                triggered = _check_sl_tp_exits(
                    broker, "jp", "paper", MagicMock(), stats, InMemoryTradeDiffSink()
                )

        self.assertEqual(triggered, {"7203"})
        self.assertEqual(stats["errors"], 0)


if __name__ == "__main__":
    unittest.main()
