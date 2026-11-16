"""Discord の約定・発注の知らせの1件ごとの明細（#1876）"""

from unittest.mock import patch

import pytest

from src.reporting.discord import discord_utils
from src.reporting.discord.trade_details import (
    TRADE_DETAIL_FIELD_NAME,
    TradeLine,
    build_trade_detail_value,
    format_trade_line,
    safe_trade_detail_field,
)

_NAMES = {"7203": "トヨタ自動車", "6758": "ソニーグループ"}


def _fake_label(market, symbol):
    name = _NAMES.get(symbol)
    return f"{symbol} {name}" if name else str(symbol)


@pytest.fixture(autouse=True)
def _labels():
    with patch("src.reporting.discord.trade_details.symbol_label", side_effect=_fake_label):
        yield


class TestFormatTradeLine:
    def test_limit_buy_shows_name_side_price_qty(self):
        line = TradeLine(symbol="7203", side=1, qty=100, price=2850.0)
        assert format_trade_line(line) == "• 7203 トヨタ自動車 買い 100株 @ 2,850円"

    def test_each_side_label(self):
        labels = [
            format_trade_line(TradeLine(symbol="6758", side=s, qty=200, price=3000.5))
            for s in (1, 2, 3, 4)
        ]
        assert labels == [
            "• 6758 ソニーグループ 買い 200株 @ 3,000.5円",
            "• 6758 ソニーグループ 売り 200株 @ 3,000.5円",
            "• 6758 ソニーグループ 空売り 200株 @ 3,000.5円",
            "• 6758 ソニーグループ 買い戻し 200株 @ 3,000.5円",
        ]

    def test_market_order_shows_nariyuki(self):
        line = TradeLine(symbol="7203", side=2, qty=100, price=0.0, is_market=True)
        assert format_trade_line(line) == "• 7203 トヨタ自動車 売り 100株 @ 成行"

    def test_unknown_name_shows_code_only(self):
        line = TradeLine(symbol="9999", side=1, qty=100, price=500.0)
        assert format_trade_line(line) == "• 9999 買い 100株 @ 500円"


class TestBuildTradeDetailValue:
    def test_empty_is_none(self):
        assert build_trade_detail_value([]) is None

    def test_all_lines_when_small(self):
        lines = [
            TradeLine(symbol="7203", side=1, qty=100, price=2850.0),
            TradeLine(symbol="6758", side=2, qty=200, price=0.0, is_market=True),
        ]
        assert build_trade_detail_value(lines) == (
            "• 7203 トヨタ自動車 買い 100株 @ 2,850円\n• 6758 ソニーグループ 売り 200株 @ 成行"
        )

    def test_over_max_lines_folds_rest(self):
        lines = [TradeLine(symbol=str(1000 + i), side=1, qty=100, price=100.0) for i in range(5)]
        value = build_trade_detail_value(lines, max_lines=3)
        assert value is not None
        rows = value.split("\n")
        assert rows[:3] == [
            "• 1000 買い 100株 @ 100円",
            "• 1001 買い 100株 @ 100円",
            "• 1002 買い 100株 @ 100円",
        ]
        assert rows[3] == "…ほか 2 件"

    def test_over_char_limit_folds_rest_within_limit(self):
        lines = [TradeLine(symbol=str(1000 + i), side=1, qty=100, price=100.0) for i in range(100)]
        value = build_trade_detail_value(lines, max_lines=100, limit=1024)
        assert value is not None
        assert len(value) <= 1024
        rows = value.split("\n")
        shown = len(rows) - 1
        assert rows[-1] == f"…ほか {100 - shown} 件"
        assert 0 < shown < 100


class TestSafeTradeDetailField:
    def test_field_shape(self):
        field = safe_trade_detail_field([TradeLine(symbol="7203", side=1, qty=100, price=1.0)])
        assert field == {
            "name": TRADE_DETAIL_FIELD_NAME,
            "value": "• 7203 トヨタ自動車 買い 100株 @ 1円",
            "inline": False,
        }

    def test_broken_detail_returns_none(self):
        assert safe_trade_detail_field([object()]) is None  # type: ignore[list-item]

    def test_none_returns_none(self):
        assert safe_trade_detail_field(None) is None


def _sent_fields(mock_send):
    return mock_send.call_args.args[1]


class TestSendWithDetails:
    def test_settle_completion_keeps_count_and_adds_detail(self):
        with patch.object(discord_utils, "send_status_fields", return_value=True) as send:
            discord_utils.send_daily_settle_completion(
                settled_count=1, details=[TradeLine(symbol="7203", side=1, qty=100, price=2850.0)]
            )
        fields = _sent_fields(send)
        names = [f["name"] for f in fields]
        assert "📊 約定件数" in names
        detail = next(f for f in fields if f["name"] == TRADE_DETAIL_FIELD_NAME)
        assert detail["value"] == "• 7203 トヨタ自動車 買い 100株 @ 2,850円"

    def test_settle_completion_broken_detail_sends_count_only(self):
        with patch.object(discord_utils, "send_status_fields", return_value=True) as send:
            assert discord_utils.send_daily_settle_completion(
                settled_count=1, details=[object()]  # type: ignore[list-item]
            )
        names = [f["name"] for f in _sent_fields(send)]
        assert names == ["🕐 時刻", "📊 約定件数"]

    def test_order_completion_adds_detail(self):
        with patch.object(discord_utils, "send_status_fields", return_value=True) as send:
            discord_utils.send_daily_order_completion(
                buy_orders=1,
                sell_orders=0,
                details=[TradeLine(symbol="7203", side=1, qty=100, price=0.0, is_market=True)],
            )
        fields = _sent_fields(send)
        assert any(f["name"] == "🟢 買い注文" for f in fields)
        detail = next(f for f in fields if f["name"] == TRADE_DETAIL_FIELD_NAME)
        assert detail["value"] == "• 7203 トヨタ自動車 買い 100株 @ 成行"

    def test_order_completion_broken_detail_sends_counts(self):
        with patch.object(discord_utils, "send_status_fields", return_value=True) as send:
            assert discord_utils.send_daily_order_completion(
                buy_orders=1, sell_orders=2, details=[object()]  # type: ignore[list-item]
            )
        names = [f["name"] for f in _sent_fields(send)]
        assert TRADE_DETAIL_FIELD_NAME not in names
        assert "🟢 買い注文" in names and "🔴 売り注文" in names


class TestCallerConversion:
    def test_settled_rows_to_lines(self):
        from src.orchestration.jobs.trade_lines import settled_trade_lines

        lines = settled_trade_lines(
            [{"order_id": "a", "symbol": "7203", "side": 1, "fill_price": 2850.0, "qty": 100}]
        )
        assert lines == [TradeLine(symbol="7203", side=1, qty=100, price=2850.0)]

    def test_notify_settle_broken_rows_sends_count_only(self):
        from src.orchestration.jobs.settle_notify import notify_settle_result

        with patch("src.reporting.discord.discord_utils.send_daily_settle_completion") as send:
            notify_settle_result(2, 0, True, settled=[{"symbol": "7203"}, {}])
        send.assert_called_once_with(settled_count=2, details=None)

    def test_notify_settle_passes_details(self):
        from src.orchestration.jobs.settle_notify import notify_settle_result

        with patch("src.reporting.discord.discord_utils.send_daily_settle_completion") as send:
            notify_settle_result(
                1,
                0,
                True,
                settled=[{"symbol": "7203", "side": 2, "fill_price": 100.0, "qty": 100}],
            )
        send.assert_called_once_with(
            settled_count=1, details=[TradeLine(symbol="7203", side=2, qty=100, price=100.0)]
        )

    def test_order_details_to_lines(self):
        from src.orchestration.jobs.trade_lines import order_trade_lines

        lines = order_trade_lines(
            {
                "order_details": [
                    {"symbol": "7203", "side": 1, "qty": 100, "price": 0.0, "is_market": True},
                    {"symbol": "6758", "side": 2, "qty": 200, "price": 3000.0, "is_market": False},
                ]
            }
        )
        assert lines == [
            TradeLine(symbol="7203", side=1, qty=100, price=0.0, is_market=True),
            TradeLine(symbol="6758", side=2, qty=200, price=3000.0, is_market=False),
        ]

    def test_order_details_broken_is_none(self):
        from src.orchestration.jobs.trade_lines import order_trade_lines

        assert order_trade_lines({"order_details": [{"symbol": "7203"}]}) is None
        assert order_trade_lines({}) is None


class TestNoteOrder:
    def test_note_order_records_detail(self):
        from src.trading.brokers.base import OrderSide, OrderType
        from src.trading.execution.stats import _new_stats, note_order

        stats = _new_stats()
        note_order(stats, "7203", OrderSide.BUY, 100, OrderType.LIMIT, 2850.0)
        note_order(stats, "6758", OrderSide.SELL, 200, OrderType.MARKET, 0.0)
        note_order(stats, "9984", OrderSide.SHORT, 100, OrderType.MARKET, 0.0)
        note_order(stats, "9983", OrderSide.SHORT_COVER, 100, OrderType.LIMIT, 50000.0)
        assert (stats["buy_orders"], stats["sell_orders"], stats["short_orders"]) == (1, 1, 2)
        assert stats["order_details"][:2] == [
            {"symbol": "7203", "side": 1, "qty": 100, "price": 2850.0, "is_market": False},
            {"symbol": "6758", "side": 2, "qty": 200, "price": 0.0, "is_market": True},
        ]
