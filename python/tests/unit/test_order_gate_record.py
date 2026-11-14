"""
ユニットテスト: 8:50 の発注の回ごとに、新規を止めた判定を JSON に書く（#1511）

損失上限・相関のゲートで止めた日も止めなかった日も results/orders/<JST の日付>.json に
止めたか・理由・reason_code・threshold_scale を残す。書き込みが落ちても発注は変わらない。
"""

import json
import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from src.infrastructure.in_memory import (
    InMemoryOrderRunSink,
    InMemoryPredictionRepository,
    InMemoryTradeDiffSink,
)
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.execution import run_daily_orders
from src.trading.types import CorrelationGateResult, TradingGateStatus
from src.utils.data_path_utils import get_results_dir

_PREDICTIONS = InMemoryPredictionRepository()
_NOW = datetime(2026, 10, 28, 8, 50, tzinfo=ZoneInfo("Asia/Tokyo"))

_ALLOWED = TradingGateStatus(
    is_allowed=True, stop_active=False, daily_loss=1_000.0, daily_loss_limit=15_630.0
)
_LOSS_LIMIT = TradingGateStatus(
    is_allowed=False,
    stop_active=True,
    reason_code="daily_loss_limit",
    reason="日次損失上限に到達: 損失=30500円 / 上限=15630円",
    daily_loss=30_500.0,
    daily_loss_limit=15_630.0,
)
_CORR_OK = CorrelationGateResult(
    is_allowed=True, enc=3.0, enc_threshold=2.0, avg_correlation=0.1, n_symbols=2
)
_CORR_BLOCKED = CorrelationGateResult(
    is_allowed=False,
    enc=1.2,
    enc_threshold=2.0,
    avg_correlation=0.8,
    n_symbols=2,
    reason="ENC 1.20 < 2.00",
)


_HELD = {"symbol": "9000", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}
_UNPRICED = {"symbol": "7203", "qty": 100, "avg_price": 0.0, "current_price": 1000.0}
_NAN_SCORE = {"symbol": "7200", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}
_NO_ROW = {"symbol": "3350", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}


def _make_broker(positions=(_HELD,)):
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 1_000_000.0
    broker.get_positions.return_value = [dict(p) for p in positions]
    broker.get_short_positions = MagicMock(return_value=[])
    broker.send_order.return_value = {"order_id": "test-001", "status": "pending"}
    return broker


def _predictions():
    return pd.DataFrame(
        [
            {
                "market": "jp",
                "symbol": symbol,
                "predicted_at": "20261028_083000",
                "current_price": 1000.0,
                "diff_ratio": diff,
                "confidence_ratio": 1.0,
            }
            for symbol, diff in (("9000", -0.03), ("7200", 0.03))
        ]
    )


def _record_path():
    return os.path.join(get_results_dir(), "orders", "2026-10-28.json")


class TestOrderGateRecord(unittest.TestCase):
    def _run(
        self,
        gate_status,
        corr_gate,
        extra_patches=(),
        positions=(_HELD,),
        predictions=None,
        broker=None,
        stop_hit=False,
    ):
        sl_risk = MagicMock()
        sl_risk.return_value.check_sl_tp.return_value = (stop_hit, False, "SL" if stop_hit else "")
        frame = _predictions() if predictions is None else predictions
        patches = [
            patch.object(_PREDICTIONS, "get_latest_by_market", return_value=frame),
            patch("src.trading.execution.runner._record_order"),
            patch("src.trading.execution.sl_tp._record_order"),
            patch(
                "src.trading.execution.runner._choose_order_params",
                return_value=(OrderType.MARKET, 0.0, "market", "open"),
            ),
            patch("src.trading.execution.sl_tp.RiskManager", sl_risk),
            patch(
                "src.trading.execution.RiskManager.evaluate_trading_gate",
                return_value=gate_status,
            ),
            patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
            patch("src.trading.execution.RiskManager.update_peak_balance"),
            patch("src.trading.execution.runner.evaluate_correlation_gate", return_value=corr_gate),
            patch("src.trading.execution.runner._load_exit_model", return_value=None),
            patch("src.trading.execution.params.get_optimal_params", return_value={}),
            patch(
                "src.trading.execution.selection.get_symbol_sector",
                side_effect=lambda market, symbol: f"UNKNOWN:{market}:{symbol}",
            ),
            patch("src.trading.execution.gate_record.now_jst", return_value=_NOW),
            *extra_patches,
        ]
        for p in patches:
            p.start()
        broker = broker or _make_broker(positions)
        try:
            stats = run_daily_orders(
                broker,
                order_run_sink=InMemoryOrderRunSink(),
                trade_diff_sink=InMemoryTradeDiffSink(),
                prediction_repo=_PREDICTIONS,
                market="jp",
                mode="paper",
            )
        finally:
            for p in reversed(patches):
                p.stop()
        return stats, broker

    @staticmethod
    def _read():
        with open(_record_path(), encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _sent(broker):
        return [(c.args[0], c.args[1]) for c in broker.send_order.call_args_list]

    def test_loss_limit_day_is_recorded(self):
        """損失上限で止めた日は、止めたか・理由・reason_code・損失と上限・scale が入る"""
        stats, _ = self._run(_LOSS_LIMIT, _CORR_OK)
        record = self._read()
        self.assertEqual(record["date"], "2026-10-28")
        self.assertTrue(record["trading_stopped"])
        self.assertEqual(record["stop_reason"], _LOSS_LIMIT.reason)
        self.assertEqual(record["reason_code"], "daily_loss_limit")
        self.assertEqual(record["daily_loss"], 30_500.0)
        self.assertEqual(record["daily_loss_limit"], 15_630.0)
        self.assertFalse(record["correlation_blocked"])
        self.assertEqual(record["threshold_scale"], stats["threshold_scale"])
        self.assertIsInstance(record["threshold_scale"], float)
        self.assertEqual(record["buy_orders"], 0)
        self.assertEqual(record["sell_orders"], 1)

    def test_correlation_blocked_day_is_recorded(self):
        """相関で止めた日は、correlation_blocked・enc・avg_correlation と scale が入る"""
        stats, _ = self._run(_ALLOWED, _CORR_BLOCKED)
        record = self._read()
        self.assertFalse(record["trading_stopped"])
        self.assertIsNone(record["stop_reason"])
        self.assertIsNone(record["reason_code"])
        self.assertTrue(record["correlation_blocked"])
        self.assertEqual(record["enc"], 1.2)
        self.assertEqual(record["avg_correlation"], 0.8)
        self.assertEqual(record["threshold_scale"], stats["threshold_scale"])
        self.assertIsInstance(record["threshold_scale"], float)
        self.assertEqual(record["buy_orders"], 0)

    def test_allowed_day_is_recorded(self):
        """止めなかった日も、止めていないことと scale・注文の件数が入る"""
        stats, _ = self._run(_ALLOWED, _CORR_OK)
        record = self._read()
        self.assertFalse(record["trading_stopped"])
        self.assertIsNone(record["stop_reason"])
        self.assertIsNone(record["reason_code"])
        self.assertFalse(record["correlation_blocked"])
        self.assertIsNone(record["stale_prediction_reason"])
        self.assertEqual(record["threshold_scale"], stats["threshold_scale"])
        self.assertIsInstance(record["threshold_scale"], float)
        self.assertEqual(record["buy_orders"], 1)
        self.assertEqual(record["sell_orders"], 1)
        self.assertEqual(record["mode"], "paper")
        self.assertIsNone(record["entry_error"])

    def test_unevaluable_holdings_are_recorded_on_prediction_day(self):
        """予測のある日に SL/TP を評価できなかった保有の銘柄と数が入る（#1516）"""
        stats, _ = self._run(_ALLOWED, _CORR_OK, positions=(_HELD, _UNPRICED))
        record = self._read()
        self.assertEqual(stats["sl_tp_unevaluable_symbols"], ["7203"])
        self.assertEqual(record["sl_tp_unevaluable_symbols"], ["7203"])
        self.assertEqual(record["sl_tp_unevaluable_count"], 1)

    def test_unevaluable_holdings_are_recorded_on_exits_only_day(self):
        """予測の無い SL/TP だけの日も、評価できなかった保有の銘柄と数が入る（#1516）"""
        stats, _ = self._run(
            _ALLOWED, _CORR_OK, positions=(_HELD, _UNPRICED), predictions=pd.DataFrame()
        )
        record = self._read()
        self.assertEqual(stats["sl_tp_unevaluable_symbols"], ["7203"])
        self.assertEqual(record["sl_tp_unevaluable_symbols"], ["7203"])
        self.assertEqual(record["sl_tp_unevaluable_count"], 1)
        self.assertEqual(record["buy_orders"], 0)

    def test_no_unevaluable_holdings_records_empty_list_and_zero(self):
        """評価できない保有が無い日は、空の一覧と 0 が入る"""
        for predictions in (None, pd.DataFrame()):
            with self.subTest(exits_only=predictions is not None):
                self._run(_ALLOWED, _CORR_OK, predictions=predictions)
                record = self._read()
                self.assertEqual(record["sl_tp_unevaluable_symbols"], [])
                self.assertEqual(record["sl_tp_unevaluable_count"], 0)

    def test_write_failure_keeps_orders_and_stats(self):
        """書き込みが落ちても、戻り値と注文は書けた日と同じ"""
        stats_ok, broker_ok = self._run(_ALLOWED, _CORR_OK)
        failing = patch(
            "src.trading.execution.gate_record._write_record", side_effect=OSError("disk full")
        )
        stats_ng, broker_ng = self._run(_ALLOWED, _CORR_OK, extra_patches=(failing,))
        self.assertEqual(stats_ng, stats_ok)
        self.assertEqual(self._sent(broker_ng), self._sent(broker_ok))
        self.assertIn(("7200", OrderSide.BUY), self._sent(broker_ng))

    def test_entry_error_day_is_recorded_before_raise(self):
        """相関ゲートが落ちて投げ直す日も、投げる前に書く"""
        failing = patch(
            "src.trading.execution.runner.evaluate_correlation_gate",
            side_effect=RuntimeError("corr down"),
        )
        with self.assertRaises(RuntimeError):
            self._run(_ALLOWED, _CORR_OK, extra_patches=(failing,))
        record = self._read()
        self.assertTrue(record["correlation_blocked"])
        self.assertIn("corr down", record["entry_error"])
        self.assertEqual(record["buy_orders"], 0)

    def test_held_without_prediction_is_recorded_and_warned(self):
        """行の無い保有と統合スコアが NaN の保有は、銘柄と数が記録に入り WARNING が出る（#1530）"""
        frame = _predictions()
        frame.loc[frame["symbol"] == "7200", "diff_ratio"] = float("nan")
        with patch("src.trading.execution.gate_record.logger") as log:
            stats, _ = self._run(
                _ALLOWED, _CORR_OK, positions=(_HELD, _NAN_SCORE, _NO_ROW), predictions=frame
            )
        record = self._read()
        self.assertEqual(stats["held_without_prediction_symbols"], ["3350", "7200"])
        self.assertEqual(record["held_without_prediction_symbols"], ["3350", "7200"])
        self.assertEqual(record["held_without_prediction_count"], 2)
        log.warning.assert_called_once()
        self.assertIn("3350", str(log.warning.call_args))
        self.assertIn("7200", str(log.warning.call_args))

    def test_all_held_predicted_records_empty_list_and_zero(self):
        """該当が無い日は、空の一覧と 0 が入り WARNING は出ない（#1530）"""
        with patch("src.trading.execution.gate_record.logger") as log:
            self._run(_ALLOWED, _CORR_OK)
        record = self._read()
        self.assertEqual(record["held_without_prediction_symbols"], [])
        self.assertEqual(record["held_without_prediction_count"], 0)
        log.warning.assert_not_called()

    def test_empty_predictions_records_null(self):
        """予測が空の日は判定をしていないので null が入る（#1530）"""
        self._run(_ALLOWED, _CORR_OK, predictions=pd.DataFrame())
        record = self._read()
        self.assertIsNone(record["held_without_prediction_symbols"])
        self.assertIsNone(record["held_without_prediction_count"])

    def test_empty_predictions_records_stale_reason(self):
        """予測が空の日は理由が記録に入り、ゲートの判定は変えない（#1572）"""
        stats, _ = self._run(_ALLOWED, _CORR_OK, predictions=pd.DataFrame())
        record = self._read()
        self.assertIn("production の予想の行が0件", stats["stale_prediction_reason"])
        self.assertEqual(record["stale_prediction_reason"], stats["stale_prediction_reason"])
        self.assertFalse(record["trading_stopped"])
        self.assertIsNone(record["stop_reason"])

    def test_read_failure_day_is_recorded_before_same_raise(self):
        """予測か保有を読めない日も、SL/TP を1回出して記録を書き、同じ例外を投げる（#1530）"""
        error = RuntimeError("read down")
        sources = {
            "predictions": patch.object(_PREDICTIONS, "get_latest_by_market", side_effect=error),
            "holdings": patch("src.trading.execution.runner._get_held_symbols", side_effect=error),
        }
        for name, failing in sources.items():
            with self.subTest(failing=name):
                if os.path.exists(_record_path()):
                    os.remove(_record_path())
                broker = _make_broker()
                with self.assertRaises(RuntimeError) as caught:
                    self._run(
                        _ALLOWED, _CORR_OK, extra_patches=(failing,), broker=broker, stop_hit=True
                    )
                self.assertIs(caught.exception, error)
                self.assertEqual(self._sent(broker), [("9000", OrderSide.SELL)])
                record = self._read()
                self.assertIn("read down", record["entry_error"])
                self.assertEqual(record["sell_orders"], 1)
                self.assertIsNone(record["held_without_prediction_symbols"])
                self.assertIsNone(record["held_without_prediction_count"])

    def _run_failing(self, *failing, broker=None, predictions=None):
        """前の記録を消して例外で落ちる朝を流し、投げられた例外と broker を返す（#1548）"""
        if os.path.exists(_record_path()):
            os.remove(_record_path())
        broker = broker or _make_broker()
        with self.assertRaises(RuntimeError) as caught:
            self._run(
                _ALLOWED, _CORR_OK, extra_patches=failing, broker=broker, predictions=predictions
            )
        return caught.exception, broker

    def test_sl_tp_step_error_is_recorded_before_same_raise(self):
        """_check_sl_tp_exits が投げる朝も予想の売りは出し、記録を書いて同じ例外で落ちる（#1548・#1731）"""
        error = RuntimeError("sl tp down")
        failing = patch("src.trading.execution.runner._check_sl_tp_exits", side_effect=error)
        caught, broker = self._run_failing(failing)
        self.assertIs(caught, error)
        self.assertEqual(self._sent(broker), [("9000", OrderSide.SELL)])
        record = self._read()
        self.assertEqual(record["entry_error"], repr(error))
        self.assertIsNone(record["token_error"])
        self.assertFalse(record["trading_stopped"])
        self.assertEqual(record["daily_loss"], 1_000.0)
        self.assertEqual(record["sell_orders"], 1)
        self.assertEqual(record["buy_orders"], 0)

    def test_short_read_error_keeps_sl_tp_and_signal_sells(self):
        """空売りを読めない朝も、SL/TP の売りと予想の売りを出し、新規は出さずに同じ例外で落ちる（#1731）"""
        stop = {"symbol": "8000", "qty": 100, "avg_price": 2000.0, "current_price": 1000.0}
        broker = _make_broker(positions=(stop, _HELD))
        error = RuntimeError("short read down")
        broker.get_short_positions.side_effect = error
        sl_risk = MagicMock()
        sl_risk.return_value.check_sl_tp.side_effect = lambda avg, cur, **kw: (
            avg > cur * 1.5,
            False,
            "SL",
        )
        sl_line = patch("src.trading.execution.sl_tp.RiskManager", sl_risk)
        caught, _ = self._run_failing(sl_line, broker=broker)
        self.assertIs(caught, error)
        self.assertEqual(self._sent(broker), [("8000", OrderSide.SELL), ("9000", OrderSide.SELL)])
        record = self._read()
        self.assertEqual(record["entry_error"], repr(error))
        self.assertEqual(record["sell_orders"], 2)
        self.assertEqual(record["buy_orders"], 0)
        self.assertEqual(record["short_orders"], 0)

    def test_short_read_in_sl_tp_error_is_recorded(self):
        """SL/TP の中の空売りの読み込みの例外（#1451）は投げたまま、記録を書く（#1548）"""
        broker = _make_broker()
        error = RuntimeError("short read down")
        broker.get_short_positions.side_effect = error
        caught, _ = self._run_failing(broker=broker)
        self.assertIs(caught, error)
        self.assertEqual(self._read()["entry_error"], repr(error))

    def test_short_positions_error_after_sells_records_sell_count(self):
        """売りを出した後に read_short_positions が投げる朝は、出した売りの数も入る（#1548）"""
        error = RuntimeError("short positions down")
        failing = patch("src.trading.execution.runner.read_short_positions", side_effect=error)
        caught, broker = self._run_failing(failing)
        self.assertIs(caught, error)
        self.assertEqual(self._sent(broker), [("9000", OrderSide.SELL)])
        record = self._read()
        self.assertEqual(record["entry_error"], repr(error))
        self.assertEqual(record["sell_orders"], 1)
        self.assertEqual(record["buy_orders"], 0)

    def test_sector_limit_error_after_sells_records_sell_count(self):
        """売りの後に業種の上限の手順が投げる朝も、出した売りの数が入る（#1548）"""
        error = RuntimeError("sector down")
        # 業種を引く手順は _select_buy_signals の中にある（#1759 で業種の上限は回す側へ移した）
        failing = patch("src.trading.execution.runner._select_buy_signals", side_effect=error)
        caught, broker = self._run_failing(failing)
        self.assertIs(caught, error)
        self.assertEqual(self._sent(broker), [("9000", OrderSide.SELL)])
        record = self._read()
        self.assertEqual(record["entry_error"], repr(error))
        self.assertEqual(record["sell_orders"], 1)

    def test_exits_only_error_is_recorded(self):
        """予測が空の朝も予測を読めない朝も、SL/TP の例外を書いて同じ例外で落ちる（#1548）"""
        read_down = patch.object(
            _PREDICTIONS, "get_latest_by_market", side_effect=RuntimeError("read down")
        )
        for name, extra, frame in (
            ("empty", (), pd.DataFrame()),
            ("read_failure", (read_down,), None),
        ):
            with self.subTest(path=name):
                error = RuntimeError("exits down")
                failing = patch("src.trading.execution.sl_tp._check_sl_tp_exits", side_effect=error)
                caught, _ = self._run_failing(failing, *extra, predictions=frame)
                self.assertIs(caught, error)
                self.assertEqual(self._read()["entry_error"], repr(error))

    def test_record_is_written_once_per_run(self):
        """投げ直しの道でも手順の例外の道でも、記録は1回だけ書く（#1548）"""
        step_error = patch(
            "src.trading.execution.runner.read_short_positions",
            side_effect=RuntimeError("short positions down"),
        )
        rethrow = patch(
            "src.trading.execution.runner.evaluate_correlation_gate",
            side_effect=RuntimeError("corr down"),
        )
        for name, failing in (("step_error", step_error), ("rethrow", rethrow)):
            with self.subTest(path=name):
                with patch("src.trading.execution.gate_record._write_record") as write:
                    self._run_failing(failing)
                self.assertEqual(write.call_count, 1)

    def test_write_failure_keeps_original_exception_and_is_not_retried(self):
        """書き込みが落ちても元の例外がそのまま投げられ、書き直さない（#1548）"""
        targets = (
            "src.trading.execution.runner.read_short_positions",
            "src.trading.execution.runner.evaluate_correlation_gate",
        )
        for target in targets:
            with self.subTest(target=target):
                error = RuntimeError("step down")
                with patch(
                    "src.trading.execution.gate_record._write_record",
                    side_effect=OSError("disk full"),
                ) as write:
                    caught, _ = self._run_failing(patch(target, side_effect=error))
                self.assertIs(caught, error)
                self.assertEqual(write.call_count, 1)


if __name__ == "__main__":
    unittest.main()
