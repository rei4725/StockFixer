"""paper/fill.py の limit_fill_price と record_trade_diff の単体テスト（#2204）"""

from unittest.mock import MagicMock, patch

import pytest

from src.trading.brokers.base import OrderSide
from src.trading.brokers.paper.fill import limit_fill_price, record_trade_diff


class TestLimitFillPrice:
    """4つの向きで、届いたときの値段と届かないときの None を確かめる"""

    @pytest.mark.parametrize(
        ("open_price", "expected"),
        [(1000.0, 990.0), (980.0, 980.0)],
    )
    def test_buy_reached_takes_min_of_limit_and_open(
        self, open_price: float, expected: float
    ) -> None:
        got = limit_fill_price(int(OrderSide.BUY), 990.0, open_price, 1010.0, 970.0)
        assert got == expected

    def test_buy_not_reached_returns_none(self) -> None:
        assert limit_fill_price(int(OrderSide.BUY), 960.0, 1000.0, 1010.0, 970.0) is None

    @pytest.mark.parametrize(
        ("open_price", "expected"),
        [(1000.0, 1005.0), (1008.0, 1008.0)],
    )
    def test_sell_reached_takes_max_of_limit_and_open(
        self, open_price: float, expected: float
    ) -> None:
        got = limit_fill_price(int(OrderSide.SELL), 1005.0, open_price, 1010.0, 970.0)
        assert got == expected

    def test_sell_not_reached_returns_none(self) -> None:
        assert limit_fill_price(int(OrderSide.SELL), 1020.0, 1000.0, 1010.0, 970.0) is None

    @pytest.mark.parametrize(
        ("open_price", "expected"),
        [(1000.0, 1005.0), (1008.0, 1008.0)],
    )
    def test_short_reached_takes_max_of_limit_and_open(
        self, open_price: float, expected: float
    ) -> None:
        got = limit_fill_price(int(OrderSide.SHORT), 1005.0, open_price, 1010.0, 970.0)
        assert got == expected

    def test_short_not_reached_returns_none(self) -> None:
        assert limit_fill_price(int(OrderSide.SHORT), 1020.0, 1000.0, 1010.0, 970.0) is None

    @pytest.mark.parametrize(
        ("open_price", "expected"),
        [(1000.0, 990.0), (980.0, 980.0)],
    )
    def test_short_cover_reached_takes_min_of_limit_and_open(
        self, open_price: float, expected: float
    ) -> None:
        got = limit_fill_price(int(OrderSide.SHORT_COVER), 990.0, open_price, 1010.0, 970.0)
        assert got == expected

    def test_short_cover_not_reached_returns_none(self) -> None:
        assert limit_fill_price(int(OrderSide.SHORT_COVER), 960.0, 1000.0, 1010.0, 970.0) is None

    def test_touching_the_edge_counts_as_reached(self) -> None:
        assert limit_fill_price(int(OrderSide.BUY), 970.0, 1000.0, 1010.0, 970.0) == 970.0
        assert limit_fill_price(int(OrderSide.SELL), 1010.0, 1000.0, 1010.0, 970.0) == 1010.0


class TestRecordTradeDiff:
    """差分の記録を組んで渡し、失敗は投げずに受ける"""

    def test_records_diff_with_fill_price(self) -> None:
        sink = MagicMock()
        record_trade_diff(sink, ("jp", "7203", "2026-11-28", 1, 1000.0, "oid1"), 995.0)
        sink.record.assert_called_once()
        diff = sink.record.call_args[0][0]
        assert diff.market == "jp"
        assert diff.symbol == "7203"
        assert diff.predicted_at == "2026-11-28"
        assert diff.side == 1
        assert diff.signal_price == 1000.0
        assert diff.mode == "paper"
        assert diff.order_id == "oid1"
        assert diff.actual_price == 995.0

    @pytest.mark.parametrize(
        ("market", "predicted_at", "signal_price"),
        [(None, "2026-11-28", 1000.0), ("jp", None, 1000.0), ("jp", "2026-11-28", None)],
    )
    def test_skips_order_without_prediction(
        self, market: str | None, predicted_at: str | None, signal_price: float | None
    ) -> None:
        sink = MagicMock()
        record_trade_diff(sink, (market, "7203", predicted_at, 1, signal_price, "oid1"), 995.0)
        sink.record.assert_not_called()

    def test_sink_failure_is_swallowed(self) -> None:
        sink = MagicMock()
        sink.record.side_effect = RuntimeError("db down")
        with patch("src.trading.brokers.paper.fill.logger") as log:
            record_trade_diff(sink, ("jp", "7203", "2026-11-28", 1, 1000.0, "oid1"), 995.0)
        sink.record.assert_called_once()
        log.error.assert_called_once()
        assert log.error.call_args[0][0] == "[paper] 約定の差分の記録に失敗: oid1"
