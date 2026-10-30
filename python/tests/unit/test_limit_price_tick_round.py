"""_choose_order_params の指値が呼値の単位に乗ることを確かめる（#1396）。"""

import unittest

import pandas as pd

from config.settings import LIMIT_ORDER_PRICE_BUFFER
from src.trading.brokers.base import OrderSide, OrderType
from src.trading.execution.params import _choose_order_params

# 東証の一般の呼値の単位（TOPIX500 以外の表）。(この値段以下, 単位)
_GENERAL_TICKS = [
    (3_000, 1),
    (5_000, 5),
    (30_000, 10),
    (50_000, 50),
    (300_000, 100),
    (500_000, 500),
    (3_000_000, 1_000),
]

_BAND_EDGE_PRICES = [2999.0, 3001.0, 4999.0, 5001.0, 29999.0, 30001.0, 49999.0, 50001.0]


def _general_tick(price: float) -> int:
    for upper, tick in _GENERAL_TICKS:
        if price <= upper:
            return tick
    raise AssertionError(f"表の外の値段: {price}")


def _is_multiple(price: float, unit: float) -> bool:
    ratio = price / unit
    return abs(ratio - round(ratio)) < 1e-9


def _make_adapter(ticker: str, close: float):
    from src.infrastructure.in_memory import InMemoryMarketDataAdapter

    adapter = InMemoryMarketDataAdapter()
    # 出来高を少なくして指値の道に入れる
    adapter.set_ohlcv_data(
        ticker,
        pd.DataFrame(
            {
                "High": [close * 1.01] * 5,
                "Low": [close * 0.99] * 5,
                "Close": [close] * 5,
                "Volume": [100_000] * 5,
            }
        ),
    )
    return adapter


class TestJpLimitPriceTickRound(unittest.TestCase):
    def _limit(self, side: OrderSide, current: float) -> tuple[float, str]:
        order_type, price, _reason, session = _choose_order_params(
            "jp", "7203", side, current, market_data=_make_adapter("7203.T", current)
        )
        self.assertEqual(order_type, OrderType.LIMIT)
        return price, session

    def test_buy_side_rounds_down_to_tick(self):
        for side in (OrderSide.BUY, OrderSide.SHORT_COVER):
            for current in _BAND_EDGE_PRICES:
                with self.subTest(side=side, current=current):
                    price, session = self._limit(side, current)
                    raw = current + current * LIMIT_ORDER_PRICE_BUFFER
                    self.assertTrue(
                        _is_multiple(price, _general_tick(price)),
                        f"{price} は単位 {_general_tick(price)} の倍数でない",
                    )
                    self.assertLessEqual(price, raw)
                    self.assertGreater(price, 0.0)
                    self.assertEqual(session, "close")

    def test_sell_side_rounds_up_to_tick(self):
        for side in (OrderSide.SELL, OrderSide.SHORT):
            for current in _BAND_EDGE_PRICES:
                with self.subTest(side=side, current=current):
                    price, session = self._limit(side, current)
                    raw = current - current * LIMIT_ORDER_PRICE_BUFFER
                    self.assertTrue(
                        _is_multiple(price, _general_tick(price)),
                        f"{price} は単位 {_general_tick(price)} の倍数でない",
                    )
                    self.assertGreaterEqual(price, raw)
                    self.assertEqual(session, "close")

    def test_buy_keeps_price_above_current_within_band(self):
        # 単位より余裕の幅が広いときは、切り下げても現在値より上に残る
        price, _ = self._limit(OrderSide.BUY, 1234.0)
        self.assertEqual(price, 1235.0)

    def test_sell_keeps_price_below_current_within_band(self):
        price, _ = self._limit(OrderSide.SELL, 1234.0)
        self.assertEqual(price, 1233.0)


class TestUsLimitPriceTickRound(unittest.TestCase):
    def test_us_limit_rounds_to_cent(self):
        current = 123.456
        adapter = _make_adapter("AAPL", current)
        for side, is_buy in (
            (OrderSide.BUY, True),
            (OrderSide.SHORT_COVER, True),
            (OrderSide.SELL, False),
            (OrderSide.SHORT, False),
        ):
            with self.subTest(side=side):
                order_type, price, _reason, _session = _choose_order_params(
                    "us", "AAPL", side, current, market_data=adapter
                )
                self.assertEqual(order_type, OrderType.LIMIT)
                self.assertTrue(_is_multiple(price, 0.01), f"{price} は 0.01 の倍数でない")
                buffer = current * LIMIT_ORDER_PRICE_BUFFER
                if is_buy:
                    self.assertLessEqual(price, current + buffer)
                else:
                    self.assertGreaterEqual(price, current - buffer)


class TestMarketOrderUnchanged(unittest.TestCase):
    def test_liquid_symbol_stays_market(self):
        from src.infrastructure.in_memory import InMemoryMarketDataAdapter

        adapter = InMemoryMarketDataAdapter()
        adapter.set_ohlcv_data(
            "7203.T",
            pd.DataFrame(
                {
                    "High": [3001.0] * 5,
                    "Low": [2999.0] * 5,
                    "Close": [3000.0] * 5,
                    "Volume": [2_000_000] * 5,
                }
            ),
        )
        result = _choose_order_params("jp", "7203", OrderSide.BUY, 3000.0, market_data=adapter)
        self.assertEqual(result, (OrderType.MARKET, 0.0, "market", "open"))


if __name__ == "__main__":
    unittest.main()
