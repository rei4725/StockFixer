"""pending_short の約定待ちと約定済みの分け方を直接確かめる（#1938）

期待値はすべて手で数えた値である。
"""

from datetime import datetime

import pandas as pd

from src.trading.brokers.paper.pending_short import pending_in_row_units, split_filled_short

_NO_SPLITS = pd.Series([], dtype=float)
# 2026-03-02 に権利落ちした 1:2 の分割
_SPLIT_2X = pd.Series([2.0], index=[pd.Timestamp("2026-03-02")])


class TestPendingInRowUnits:
    def test_no_orders_returns_none(self) -> None:
        assert pending_in_row_units(None, _NO_SPLITS, datetime(2026, 3, 3)) is None
        assert pending_in_row_units([], _NO_SPLITS, datetime(2026, 3, 3)) is None

    def test_no_splits_sums_orders_as_is(self) -> None:
        # 100株 + 50株 = 150株、金額 100000 + 0 = 100000
        orders = [
            (100, 100000.0, datetime(2026, 2, 27, 1, 0)),
            (50, 0.0, datetime(2026, 2, 27, 2, 0)),
        ]
        assert pending_in_row_units(orders, _NO_SPLITS, datetime(2026, 3, 3)) == (150, 100000.0)

    def test_order_across_2x_split_doubles_qty(self) -> None:
        # 2/27 の注文 100株、行は 3/3 に直してある。間の 3/2 に 1:2 → 100 × 2 = 200株。
        # 金額は分割で変わらず 100 × 1000 = 100000
        orders = [(100, 100000.0, datetime(2026, 2, 27, 1, 0))]
        assert pending_in_row_units(orders, _SPLIT_2X, datetime(2026, 3, 3)) == (200, 100000.0)

    def test_order_after_split_is_not_scaled(self) -> None:
        # 注文が権利落ちの後（3/3）なら比は 1.0 で 100株のまま
        orders = [(100, 100000.0, datetime(2026, 3, 3, 1, 0))]
        assert pending_in_row_units(orders, _SPLIT_2X, datetime(2026, 3, 4)) == (100, 100000.0)


class TestSplitFilledShort:
    def test_no_pending_returns_all_filled(self) -> None:
        assert split_filled_short(None, 100, 100, 1000.0) == (0, 100, 1000.0)
        assert split_filled_short((0, 0.0), 100, 100, 1000.0) == (0, 100, 1000.0)

    def test_raw_qty_zero_uses_ratio_one(self) -> None:
        # 比 1.0 → 約定待ち 50株、約定済み 100 - 50 = 50株、
        # 平均単価 (1000 × 100 - 0) / 50 = 2000
        assert split_filled_short((50, 0.0), 0, 100, 1000.0) == (50, 50, 2000.0)

    def test_pending_more_than_holding_is_capped(self) -> None:
        # 約定待ち 150株 > 保有 100株 → min で 100株に切り詰め、約定済み 0株
        assert split_filled_short((150, 0.0), 100, 100, 1000.0) == (100, 0, 1000.0)

    def test_no_filled_returns_avg_as_is(self) -> None:
        # 約定待ち 100株 = 保有 100株 → 約定済み 0株、平均単価は avg の 500 のまま
        assert split_filled_short((100, 50000.0), 100, 100, 500.0) == (100, 0, 500.0)

    def test_split_ratio_scales_pending_qty(self) -> None:
        # 行は分割前 100株 → 直して 200株（比 2.0）、平均単価 1000 → 500。
        # 約定待ちの注文 40株 × 2 = 80株、約定済み 200 - 80 = 120株、
        # 金額 40 × 600 = 24000 を引いて (500 × 200 - 24000) / 120 = 76000 / 120
        pending_qty, filled_qty, filled_avg = split_filled_short((40, 24000.0), 100, 200, 500.0)
        assert (pending_qty, filled_qty) == (80, 120)
        assert filled_avg == 76000.0 / 120
