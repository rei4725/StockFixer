"""ユニットテスト: 16:30 のルールの知らせに、発注しなかった理由ごとの数を出す（#2114）

execute_rule_paper_trades は数えた skip_reasons を戻り値に足すだけで、発注の判定・順・数は
変えない。知らせは no_signal を除いた理由が1件以上あるときだけ、数の多い順の1行を足す。
"""

from unittest.mock import MagicMock, patch

from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.orchestration.jobs.rule_signal_notify import notify_rule_signals
from src.reporting.discord.notifications_rule import send_rule_daily_signals
from src.trading.brokers.base import BrokerBase
from src.trading.rule_execution import execute_rule_paper_trades

_SIGNALS = [
    {"symbol": "7203", "rule": "bollinger_band", "signal": 1, "price": 2500, "win_rate": 0.77},
    {"symbol": "6758", "rule": "rsi", "signal": -1, "price": 3000, "win_rate": 0.6},
]


def _sent_text(**kwargs) -> str:
    with (
        patch(
            "src.reporting.discord.notifications_rule.send_webhook_text", return_value=True
        ) as sent,
        patch("src.reporting.discord.notifications_rule.symbol_name_suffix", return_value=""),
        patch(
            "src.reporting.discord.notifications_rule.format_jst",
            return_value="2026/11/19 16:30 JST",
        ),
    ):
        send_rule_daily_signals(_SIGNALS, "jp", 1, 0, 100, track_record={}, **kwargs)
    return str(sent.call_args.args[0])


class TestNoticeSkipReasonsLine:
    def test_text_unchanged_without_reasons_or_with_only_no_signal(self):
        base = _sent_text()
        assert _sent_text(skip_reasons=None) == base
        assert _sent_text(skip_reasons={}) == base
        assert _sent_text(skip_reasons={"no_signal": 157}) == base
        assert "発注しなかった" not in base

    def test_line_lists_reasons_by_count_without_no_signal(self):
        text = _sent_text(
            skip_reasons={"zero_qty_budget_cap": 11, "zero_qty_under_lot": 1, "no_signal": 157}
        )
        lines = text.split("\n")
        idx = next(i for i, line in enumerate(lines) if line.startswith("ペーパー発注:"))
        assert lines[idx + 1] == "発注しなかった: zero_qty_budget_cap=11 / zero_qty_under_lot=1"
        assert "no_signal" not in text
        # 足した行のほかは、理由を渡さないときと同じ
        assert lines[: idx + 1] + lines[idx + 2 :] == _sent_text().split("\n")

    def test_zero_counts_are_not_listed(self):
        text = _sent_text(skip_reasons={"sector_cap": 0, "no_signal": 3})
        assert "発注しなかった" not in text


class TestNotifyPassesSkipReasons:
    def test_skip_reasons_are_passed_to_the_notice(self):
        stats = {"buy_orders": 1, "sell_orders": 0, "skip_reasons": {"sector_cap": 2}}
        with (
            patch("src.scoring.rule_report.load_rule_track_record", return_value={}),
            patch("src.reporting.discord.discord_utils.send_rule_daily_signals") as sent,
        ):
            notify_rule_signals([], "jp", stats, 100)
        assert sent.call_args.kwargs["skip_reasons"] == {"sector_cap": 2}

    def test_missing_skip_reasons_pass_none(self):
        with (
            patch("src.scoring.rule_report.load_rule_track_record", return_value={}),
            patch("src.reporting.discord.discord_utils.send_rule_daily_signals") as sent,
        ):
            notify_rule_signals([], "jp", {"buy_orders": 0, "sell_orders": 0}, 100)
        assert sent.call_args.kwargs["skip_reasons"] is None


def _broker() -> MagicMock:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "mock"
    broker.get_positions.return_value = []
    broker.get_short_positions = MagicMock(return_value=[])
    broker.get_pending_buy_symbols = MagicMock(return_value=set())
    broker.send_order.return_value = {"order_id": "ord-2114", "status": "pending"}
    return broker


def test_execute_returns_skip_reasons_matching_skipped():
    broker = _broker()
    risk = MagicMock()
    risk.evaluate_trading_gate.return_value = MagicMock(is_allowed=True, reason=None)
    risk.calc_position_size.return_value = 500
    risk.zero_reason = None
    corr = MagicMock(return_value=MagicMock(is_allowed=True, reason=None))
    signals = [
        {"symbol": "1111", "signal": 1, "price": 500.0},  # 予算10万で200株 → 発注
        {"symbol": "2222", "signal": 1, "price": 1500.0},  # 予算の上限で0株
        {"symbol": "3333", "signal": -1, "price": 800.0},  # 保有の無い SELL
        {"symbol": "4444", "signal": 0, "price": 900.0},  # シグナルなし
        {"symbol": "5555", "signal": 0, "price": 900.0},
    ]
    saved: list = []
    with (
        patch("src.trading.rule_execution.RiskManager", return_value=risk),
        patch("src.trading.rule_execution.evaluate_correlation_gate", corr),
        patch("src.trading.rule_execution.get_symbol_sector", return_value="A"),
        patch("src.trading.rule_execution.MAX_POSITIONS", 10),
        patch("src.trading.rule_execution.MAX_SECTOR_POSITIONS", 2),
        patch("src.trading.rule_execution.MAX_ORDERS_PER_RUN", 10),
        patch(
            "src.trading.rule_execution.get_ticker",
            side_effect=lambda market, symbol: f"{symbol}.T",
        ),
        patch(
            "src.trading.rule_execution.save_rule_gate_record",
            side_effect=lambda record, **_: saved.append(record),
        ),
    ):
        result = execute_rule_paper_trades(
            signals,
            "jp",
            trade_diff_sink=InMemoryTradeDiffSink(),
            broker=broker,
            initial_budget_per_trade=100_000,
            record_gate=True,
        )

    assert {k: result[k] for k in ("buy_orders", "sell_orders", "skipped", "errors")} == {
        "buy_orders": 1,
        "sell_orders": 0,
        "skipped": 4,
        "errors": 0,
    }
    assert result["skip_reasons"] == {
        "zero_qty_budget_cap": 1,
        "sell_not_held": 1,
        "no_signal": 2,
    }
    assert sum(result["skip_reasons"].values()) == result["skipped"]
    # 記録に写す理由は戻り値と同じで、発注は1件だけ
    assert saved[0].skip_reasons == result["skip_reasons"]
    broker.send_order.assert_called_once()
    assert broker.send_order.call_args.kwargs["symbol"] == "1111.T"
    assert broker.send_order.call_args.kwargs["qty"] == 200
