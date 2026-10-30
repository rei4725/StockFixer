"""指値の余裕の幅が少なくとも呼値の単位1つ分あることを確かめる（#1403）。"""

import unittest

import pandas as pd

from src.trading.brokers.base import OrderSide, OrderType
from src.trading.execution.params import _choose_order_params

# 東証の一般の呼値の単位（TOPIX500 以外の表）。(この値段以下, 単位)
_GENERAL_TICKS = [
    (3_000, 1),
    (5_000, 5),
    (30_000, 10),
    (50_000, 50),
    (300_000, 100),
]

# 帯の境のすぐ上。0.1% の余裕の幅が単位より小さい値段
_JUST_ABOVE_EDGE_PRICES = [3001.0, 5001.0, 30001.0, 50001.0]


def _general_tick(price: float) -> int:
    for upper, tick in _GENERAL_TICKS:
        if price <= upper:
            return tick
    raise AssertionError(f"表の外の値段: {price}")


def _is_multiple(price: float, unit: float) -> bool:
    ratio = price / unit
    return abs(ratio - round(ratio)) < 1e-9


def _limit(side: OrderSide, current: float) -> float:
    from src.infrastructure.in_memory import InMemoryMarketDataAdapter

    adapter = InMemoryMarketDataAdapter()
    # 出来高を少なくして指値の道に入れる
    adapter.set_ohlcv_data(
        "7203.T",
        pd.DataFrame(
            {
                "High": [current * 1.01] * 5,
                "Low": [current * 0.99] * 5,
                "Close": [current] * 5,
                "Volume": [100_000] * 5,
            }
        ),
    )
    order_type, price, _reason, _session = _choose_order_params(
        "jp", "7203", side, current, market_data=adapter
    )
    assert order_type == OrderType.LIMIT
    return price


class TestLimitBufferMinOneTick(unittest.TestCase):
    def test_buy_stays_at_or_above_current_at_band_edge(self):
        for side in (OrderSide.BUY, OrderSide.SHORT_COVER):
            for current in _JUST_ABOVE_EDGE_PRICES:
                with self.subTest(side=side, current=current):
                    price = _limit(side, current)
                    self.assertGreaterEqual(price, current)
                    self.assertTrue(
                        _is_multiple(price, _general_tick(price)),
                        f"{price} は単位 {_general_tick(price)} の倍数でない",
                    )

    def test_sell_stays_at_or_below_current_at_band_edge(self):
        for side in (OrderSide.SELL, OrderSide.SHORT):
            for current in _JUST_ABOVE_EDGE_PRICES:
                with self.subTest(side=side, current=current):
                    price = _limit(side, current)
                    self.assertLessEqual(price, current)
                    self.assertGreater(price, 0.0)
                    self.assertTrue(
                        _is_multiple(price, _general_tick(price)),
                        f"{price} は単位 {_general_tick(price)} の倍数でない",
                    )

    def test_buffer_wider_than_tick_keeps_price(self):
        # 0.1% が単位より広い値段では、指値は今までと変わらない
        self.assertEqual(_limit(OrderSide.BUY, 1234.0), 1235.0)
        self.assertEqual(_limit(OrderSide.SELL, 1234.0), 1233.0)
        self.assertEqual(_limit(OrderSide.BUY, 20000.0), 20020.0)
        self.assertEqual(_limit(OrderSide.SELL, 20000.0), 19980.0)


if __name__ == "__main__":
    unittest.main()
