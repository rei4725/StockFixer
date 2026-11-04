"""
ユニットテスト: 16:30 のルールの発注の回ごとに、新規を止めた判定を JSON に書く（#1514）

リスクゲート・相関ゲート・取れない銘柄が多すぎる日・判定の手順の例外の日のそれぞれで、
results/orders/rule/<JST の日付>.json に判定と止めた理由ごとの数を残す。
8:50 の記録（results/orders/<JST の日付>.json）は上書きしない。書き込みが落ちても発注は変わらない。
"""

import json
import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.orchestration.job_errors import NonRetryableJobError
from src.rule_engine.types import RuleSignalBatch, UnavailableRuleSymbol
from src.trading.brokers.base import BrokerBase
from src.trading.rule_execution import execute_rule_paper_trades
from src.trading.types import CorrelationGateResult, TradingGateStatus
from src.utils.data_path_utils import get_results_dir

_NOW = datetime(2026, 10, 27, 16, 30, tzinfo=ZoneInfo("Asia/Tokyo"))

_ALLOWED = TradingGateStatus(is_allowed=True, stop_active=False)
_LOSS_LIMIT = TradingGateStatus(
    is_allowed=False,
    stop_active=True,
    reason_code="daily_loss_limit",
    reason="日次損失上限に到達: 損失=33575円 / 上限=15630円",
    daily_loss=33_575.0,
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

# 保有 1000 の売り・新規の買い 3000・シグナル無し 4000
_SIGNALS = [
    {"symbol": "1000", "signal": -1, "price": 1000.0},
    {"symbol": "3000", "signal": 1, "price": 1000.0},
    {"symbol": "4000", "signal": 0, "price": 1000.0},
]

# 保有中で足が止まった 5202 と、保有していない 6000（シグナルを出せなかった銘柄。#1520）
_HELD_5202 = {"symbol": "5202", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}
_NG_HELD_ROW = {"symbol": "5202", "rule": "rsi_contrarian", "reason": "シグナルなし"}
_NG_NOT_HELD_ROW = {"symbol": "6000", "rule": "macd_cross", "reason": "例外"}
_UNAVAILABLE = [
    UnavailableRuleSymbol(**_NG_HELD_ROW),
    UnavailableRuleSymbol(**_NG_NOT_HELD_ROW),
]


def _broker() -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "mock"
    broker.get_positions.return_value = [
        {"symbol": "1000", "qty": 100, "avg_price": 1000.0, "current_price": 1000.0}
    ]
    broker.get_short_positions = MagicMock(return_value=[])
    broker.send_order.return_value = {"order_id": "ord-1514", "status": "pending"}
    return broker


def _rule_path() -> str:
    return os.path.join(get_results_dir(), "orders", "rule", "2026-10-27.json")


def _morning_path() -> str:
    return os.path.join(get_results_dir(), "orders", "2026-10-27.json")


def _read() -> dict:
    with open(_rule_path(), encoding="utf-8") as f:
        return json.load(f)


def _patches(gate, corr, extra=()):
    risk = MagicMock()
    if isinstance(gate, Exception):
        risk.evaluate_trading_gate.side_effect = gate
    else:
        risk.evaluate_trading_gate.return_value = gate
    risk.calc_position_size.return_value = 100
    return [
        patch("src.trading.rule_execution.RiskManager", return_value=risk),
        patch("src.trading.rule_execution.evaluate_correlation_gate", return_value=corr),
        patch("src.trading.rule_execution.get_symbol_sector", return_value="sec"),
        patch("src.trading.rule_execution.MAX_POSITIONS", 10),
        patch("src.trading.rule_execution.MAX_SECTOR_POSITIONS", 10),
        patch(
            "src.trading.rule_execution.get_ticker",
            side_effect=lambda market, symbol: f"{symbol}.T",
        ),
        patch("src.trading.execution.gate_record.now_jst", return_value=_NOW),
        *extra,
    ]


class TestRuleGateRecord(unittest.TestCase):
    def _run(
        self,
        gate,
        corr,
        *,
        extra=(),
        record_gate=True,
        allow_new_buys=True,
        pending_buys_error=None,
        positions_error=None,
        broker=None,
        **kwargs,
    ):
        broker = broker or _broker()
        broker.get_positions.return_value.append(_HELD_5202)
        if positions_error is not None:
            broker.get_positions.side_effect = positions_error
        if pending_buys_error is not None:
            broker.get_pending_buy_symbols = MagicMock(side_effect=pending_buys_error)
        patches = _patches(gate, corr, extra)
        for p in patches:
            p.start()
        try:
            result = execute_rule_paper_trades(
                _SIGNALS,
                "jp",
                trade_diff_sink=InMemoryTradeDiffSink(),
                broker=broker,
                allow_new_buys=allow_new_buys,
                record_gate=record_gate,
                **kwargs,
            )
        finally:
            for p in reversed(patches):
                p.stop()
        return result, broker

    @staticmethod
    def _sent(broker):
        return [(c.kwargs["symbol"], c.kwargs["side"]) for c in broker.send_order.call_args_list]

    def test_risk_gate_day_is_recorded(self):
        """リスクゲートで止めた日は、可否・理由と、止めた理由ごとの数が入る"""
        result, _ = self._run(_LOSS_LIMIT, _CORR_OK)
        record = _read()
        self.assertEqual(record["date"], "2026-10-27")
        self.assertEqual(record["market"], "jp")
        self.assertFalse(record["risk_gate_allowed"])
        self.assertEqual(record["risk_gate_reason"], _LOSS_LIMIT.reason)
        self.assertEqual(record["risk_gate_reason_code"], "daily_loss_limit")
        self.assertIsNone(record["correlation_gate_allowed"])
        self.assertTrue(record["allow_new_buys"])
        self.assertIsNone(record["new_buy_step_error"])
        self.assertEqual(record["skip_reasons"], {"risk_gate": 1, "no_signal": 1})
        self.assertEqual(record["buy_orders"], 0)
        self.assertEqual(record["sell_orders"], 1)
        self.assertEqual(record["skipped"], result["skipped"])
        self.assertEqual(record["errors"], 0)

    def test_correlation_gate_day_is_recorded(self):
        """相関ゲートで止めた日は、相関の可否・理由・ENC と相関で止めた数が入る"""
        self._run(_ALLOWED, _CORR_BLOCKED)
        record = _read()
        self.assertTrue(record["risk_gate_allowed"])
        self.assertFalse(record["correlation_gate_allowed"])
        self.assertEqual(record["correlation_gate_reason"], "ENC 1.20 < 2.00")
        self.assertEqual(record["enc"], 1.2)
        self.assertEqual(record["skip_reasons"], {"correlation_gate": 1, "no_signal": 1})
        self.assertEqual(record["buy_orders"], 0)

    def test_allowed_day_is_recorded(self):
        """止めなかった日も、通ったことと注文の数が入る"""
        self._run(_ALLOWED, _CORR_OK)
        record = _read()
        self.assertTrue(record["risk_gate_allowed"])
        self.assertTrue(record["correlation_gate_allowed"])
        self.assertEqual(record["skip_reasons"], {"no_signal": 1})
        self.assertEqual(record["buy_orders"], 1)
        self.assertEqual(record["sell_orders"], 1)

    def test_buy_step_error_day_is_recorded_before_raise(self):
        """新規買いの判定の手順が落ちて投げ直す日も、投げる前に書く"""
        with self.assertRaisesRegex(RuntimeError, "gate down"):
            self._run(RuntimeError("gate down"), _CORR_OK)
        record = _read()
        self.assertIsNone(record["risk_gate_allowed"])
        self.assertIn("gate down", record["new_buy_step_error"])
        self.assertEqual(record["skip_reasons"], {"new_buy_step_error": 1, "no_signal": 1})
        self.assertEqual(record["sell_orders"], 1)

    def test_pending_buy_read_failure_day_is_recorded_before_raise(self):
        """約定待ちの BUY を読めない日も、止めた BUY と保有の SELL を書いてから投げる（#1539）"""
        with self.assertRaisesRegex(RuntimeError, "pending down"):
            self._run(_ALLOWED, _CORR_OK, pending_buys_error=RuntimeError("pending down"))
        record = _read()
        self.assertIn("pending down", record["new_buy_step_error"])
        self.assertEqual(record["skip_reasons"], {"new_buy_step_error": 1, "no_signal": 1})
        self.assertEqual(record["buy_orders"], 0)
        self.assertEqual(record["sell_orders"], 1)

    def test_positions_read_failure_day_is_recorded_before_raise(self):
        """保有を読めない日も、注文を出さず、記録を書いてから同じ例外を投げる（#1549）"""
        error = RuntimeError("positions down")
        broker = _broker()
        with self.assertRaises(RuntimeError) as caught:
            self._run(_ALLOWED, _CORR_OK, positions_error=error, broker=broker)
        self.assertIs(caught.exception, error)
        broker.send_order.assert_not_called()
        record = _read()
        self.assertIn("positions down", record["new_buy_step_error"])
        self.assertIsNone(record["risk_gate_allowed"])
        self.assertIsNone(record["correlation_gate_allowed"])
        self.assertEqual(record["skip_reasons"], {})
        self.assertEqual(record["buy_orders"], 0)
        self.assertEqual(record["sell_orders"], 0)
        self.assertEqual(record["skipped"], 0)
        self.assertEqual(record["errors"], 0)

    def test_positions_read_failure_day_records_null_for_held_lists(self):
        """保有を読めない日の保有の一覧と数は null で、判定した日の空と 0 と分ける（#1549）"""
        with self.assertRaisesRegex(RuntimeError, "positions down"):
            self._run(
                _ALLOWED,
                _CORR_OK,
                positions_error=RuntimeError("positions down"),
                unavailable_symbols=_UNAVAILABLE,
            )
        record = _read()
        self.assertIsNone(record["unjudged_held_symbols"])
        self.assertIsNone(record["unjudged_held_count"])
        self.assertIsNone(record["unavailable_held_symbols"])
        self.assertIsNone(record["unavailable_held_count"])
        # 保有に関わらない、出せなかった銘柄の一覧と数はそのまま写す
        self.assertEqual(record["unavailable_symbols"], [_NG_HELD_ROW, _NG_NOT_HELD_ROW])
        self.assertEqual(record["unavailable_count"], 2)

    def test_positions_read_failure_not_recorded_unless_asked(self):
        """record_gate=False の呼び手では、保有を読めない日も書かない（#1549）"""
        with self.assertRaisesRegex(RuntimeError, "positions down"):
            self._run(
                _ALLOWED,
                _CORR_OK,
                record_gate=False,
                positions_error=RuntimeError("positions down"),
            )
        self.assertFalse(os.path.exists(_rule_path()))

    def test_positions_read_failure_write_failure_raises_original_error(self):
        """記録の書き込みが落ちても、元の例外がそのまま投げられる（#1549）"""
        error = RuntimeError("positions down")
        failing = patch(
            "src.trading.execution.gate_record._write_record", side_effect=OSError("disk full")
        )
        with self.assertLogs("src.trading.execution.gate_record", level="ERROR"):
            with self.assertRaises(RuntimeError) as caught:
                self._run(_ALLOWED, _CORR_OK, extra=(failing,), positions_error=error)
        self.assertIs(caught.exception, error)

    def test_does_not_overwrite_morning_record(self):
        """同じ日の 8:50 の記録のファイルを上書きしない"""
        os.makedirs(os.path.dirname(_morning_path()), exist_ok=True)
        with open(_morning_path(), "w", encoding="utf-8") as f:
            f.write('{"morning": true}')
        self._run(_LOSS_LIMIT, _CORR_OK)
        with open(_morning_path(), encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"morning": true}')
        self.assertTrue(os.path.exists(_rule_path()))

    def test_write_failure_keeps_orders_and_result(self):
        """書き込みが落ちても、戻り値と注文は書けた日と同じ"""
        result_ok, broker_ok = self._run(_ALLOWED, _CORR_OK)
        failing = patch(
            "src.trading.execution.gate_record._write_record", side_effect=OSError("disk full")
        )
        with self.assertLogs("src.trading.execution.gate_record", level="ERROR"):
            result_ng, broker_ng = self._run(_ALLOWED, _CORR_OK, extra=(failing,))
        self.assertEqual(result_ng, result_ok)
        self.assertEqual(set(result_ng), {"buy_orders", "sell_orders", "skipped", "errors"})
        self.assertEqual(self._sent(broker_ng), self._sent(broker_ok))

    def test_not_recorded_unless_asked(self):
        """呼び出し元が頼まないとき（手で流す CLI など）は書かない"""
        self._run(_LOSS_LIMIT, _CORR_OK, record_gate=False)
        self.assertFalse(os.path.exists(_rule_path()))

    def _assert_unavailable_recorded(self, record):
        self.assertEqual(record["unavailable_symbols"], [_NG_HELD_ROW, _NG_NOT_HELD_ROW])
        self.assertEqual(record["unavailable_count"], 2)
        self.assertEqual(record["unavailable_held_symbols"], [_NG_HELD_ROW])
        self.assertEqual(record["unavailable_held_count"], 1)

    def test_unavailable_symbols_are_recorded(self):
        """出せなかった銘柄の一覧と数が入り、保有中の銘柄は別の一覧と数に入る（#1520）"""
        result, broker = self._run(_ALLOWED, _CORR_OK, unavailable_symbols=_UNAVAILABLE)
        self._assert_unavailable_recorded(_read())
        self.assertEqual(result["sell_orders"], 1)
        self.assertEqual(result["buy_orders"], 1)
        self.assertNotIn("5202.T", [symbol for symbol, _ in self._sent(broker)])

    def test_unavailable_symbols_are_recorded_when_new_buys_are_stopped(self):
        """allow_new_buys=False の日も、一覧と数が入る（#1520）"""
        self._run(_ALLOWED, _CORR_OK, allow_new_buys=False, unavailable_symbols=_UNAVAILABLE)
        record = _read()
        self.assertFalse(record["allow_new_buys"])
        self._assert_unavailable_recorded(record)

    def test_unavailable_symbols_are_recorded_before_buy_step_error_raise(self):
        """新規買いの判定の手順が落ちて投げ直す日も、一覧と数が入る（#1520）"""
        with self.assertRaisesRegex(RuntimeError, "gate down"):
            self._run(RuntimeError("gate down"), _CORR_OK, unavailable_symbols=_UNAVAILABLE)
        record = _read()
        self.assertIn("gate down", record["new_buy_step_error"])
        self._assert_unavailable_recorded(record)

    def test_no_unavailable_day_records_empty_lists_and_zero(self):
        """出せなかった銘柄が無い日は、空の一覧と 0 が入る（#1520）"""
        self._run(_ALLOWED, _CORR_OK)
        record = _read()
        self.assertEqual(record["unavailable_symbols"], [])
        self.assertEqual(record["unavailable_count"], 0)
        self.assertEqual(record["unavailable_held_symbols"], [])
        self.assertEqual(record["unavailable_held_count"], 0)

    def test_write_failure_with_unavailable_keeps_orders_and_result(self):
        """出せなかった銘柄がある日も、書き込みが落ちて戻り値と注文は変わらない（#1520）"""
        result_ok, broker_ok = self._run(_ALLOWED, _CORR_OK, unavailable_symbols=_UNAVAILABLE)
        failing = patch(
            "src.trading.execution.gate_record._write_record", side_effect=OSError("disk full")
        )
        with self.assertLogs("src.trading.execution.gate_record", level="ERROR"):
            result_ng, broker_ng = self._run(
                _ALLOWED, _CORR_OK, extra=(failing,), unavailable_symbols=_UNAVAILABLE
            )
        self.assertEqual(result_ng, result_ok)
        self.assertEqual(self._sent(broker_ng), self._sent(broker_ok))

    def test_unjudged_holding_is_recorded(self):
        """signals にも unavailable にも入らない保有が一覧と数に入る（#1527）"""
        result, broker = self._run(_ALLOWED, _CORR_OK)
        record = _read()
        self.assertEqual(record["unjudged_held_symbols"], ["5202"])
        self.assertEqual(record["unjudged_held_count"], 1)
        self.assertEqual(record["unavailable_held_symbols"], [])
        self.assertEqual(result["sell_orders"], 1)
        self.assertNotIn("5202.T", [symbol for symbol, _ in self._sent(broker)])

    def test_no_unjudged_day_records_empty_list_and_zero(self):
        """保有がどれも signals か unavailable に入る日は、空の一覧と 0 が入る（#1527）"""
        self._run(_ALLOWED, _CORR_OK, unavailable_symbols=_UNAVAILABLE)
        record = _read()
        self.assertEqual(record["unjudged_held_symbols"], [])
        self.assertEqual(record["unjudged_held_count"], 0)
        self._assert_unavailable_recorded(record)

    def test_write_failure_with_unjudged_holding_keeps_orders_and_result(self):
        """対象に入らなかった保有がある日も、書き込みが落ちて戻り値と注文は変わらない（#1527）"""
        result_ok, broker_ok = self._run(_ALLOWED, _CORR_OK)
        self.assertEqual(_read()["unjudged_held_count"], 1)
        failing = patch(
            "src.trading.execution.gate_record._write_record", side_effect=OSError("disk full")
        )
        with self.assertLogs("src.trading.execution.gate_record", level="ERROR"):
            result_ng, broker_ng = self._run(_ALLOWED, _CORR_OK, extra=(failing,))
        self.assertEqual(result_ng, result_ok)
        self.assertEqual(self._sent(broker_ng), self._sent(broker_ok))


class TestDailyRuleSignalsGateRecord(unittest.TestCase):
    @staticmethod
    def _run_daily(batch, broker, notify=None):
        from src.orchestration.jobs import daily

        patches = _patches(_ALLOWED, _CORR_OK) + [
            patch("src.rule_engine.pipeline.generate_rule_signals", return_value=batch),
            patch("src.trading.brokers.account.factory.make_order_broker", return_value=broker),
            patch("src.infrastructure.discord_notification_adapter.DiscordNotificationAdapter"),
            patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"),
            patch("src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink"),
            patch(
                "src.reporting.discord.discord_utils.send_rule_daily_signals",
                notify or MagicMock(),
            ),
            patch("src.orchestration.jobs.daily._skip_if_jp_market_closed", return_value=False),
        ]
        for p in patches:
            p.start()
        try:
            daily.run_daily_rule_signals()
        finally:
            for p in reversed(patches):
                p.stop()

    def test_no_effective_rule_day_records_all_holdings(self):
        """有効なルールが0件の日も記録が書かれ、保有が一覧と数に入る（#1527）

        ジョブは失敗で終わるが、記録は知らせの前に書かれている（#1571）。
        """
        broker = _broker()
        broker.get_positions.return_value.append(_HELD_5202)
        held_at_notice = []
        notify = MagicMock(
            side_effect=lambda **_: held_at_notice.append(_read()["unjudged_held_count"])
        )
        with self.assertLogs("src.orchestration.jobs.unavailable_gates", level="ERROR"):
            with self.assertRaises(NonRetryableJobError):
                self._run_daily(RuleSignalBatch(), broker, notify)
        self.assertEqual(held_at_notice, [2])
        record = _read()
        self.assertEqual(record["unjudged_held_symbols"], ["1000", "5202"])
        self.assertEqual(record["unjudged_held_count"], 2)
        self.assertEqual(record["unavailable_count"], 0)
        self.assertEqual(record["sell_orders"], 0)
        broker.send_order.assert_not_called()

    def test_mostly_unavailable_day_is_recorded(self):
        """取れない銘柄が多すぎる日は、allow_new_buys=False と止めた数が入る"""
        batch = RuleSignalBatch(
            signals=list(_SIGNALS),
            unavailable=[
                UnavailableRuleSymbol(symbol=f"NG{i}", rule="rsi_contrarian", reason="データ不足")
                for i in range(4)
            ],
            total=7,
        )
        with self.assertRaises(RuntimeError):
            self._run_daily(batch, _broker())
        record = _read()
        self.assertFalse(record["allow_new_buys"])
        self.assertTrue(record["risk_gate_allowed"])
        self.assertEqual(record["skip_reasons"], {"allow_new_buys": 1, "no_signal": 1})
        self.assertEqual(record["sell_orders"], 1)
        self.assertEqual(record["buy_orders"], 0)
        # 16:30 のジョブが batch.unavailable を渡す（#1520）。NG0〜3 は保有でない
        self.assertEqual(record["unavailable_count"], 4)
        self.assertEqual(
            [row["symbol"] for row in record["unavailable_symbols"]],
            ["NG0", "NG1", "NG2", "NG3"],
        )
        self.assertEqual(record["unavailable_held_symbols"], [])
        self.assertEqual(record["unavailable_held_count"], 0)


if __name__ == "__main__":
    unittest.main()
