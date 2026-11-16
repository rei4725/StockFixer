"""domain の tick_size と round_limit_price の直接のテスト（#1920）。

期待値の呼値の表は、本番の _JP_TICK_TABLE を import せずにここに別に書く。
表の写し間違いを拾うためである。
"""

from __future__ import annotations

import pytest

from src.domain.tick_size import round_limit_price, tick_size

# 東証の一般の呼値の単位（この値段以下, 単位）。本番の表とは別に手で書いた期待値
_EXPECTED_JP_BANDS: tuple[tuple[int, int], ...] = (
    (3_000, 1),
    (5_000, 5),
    (30_000, 10),
    (50_000, 50),
    (300_000, 100),
    (500_000, 500),
    (3_000_000, 1_000),
    (5_000_000, 5_000),
    (30_000_000, 10_000),
    (50_000_000, 50_000),
)
_EXPECTED_JP_TOP = 100_000

# 帯のちょうどの境と、その1つ上（次の帯の単位）
_BOUNDARY_CASES = [(upper, tick) for upper, tick in _EXPECTED_JP_BANDS]
_ABOVE_BOUNDARY_CASES = [
    (upper + 1, next_tick)
    for (upper, _), next_tick in zip(
        _EXPECTED_JP_BANDS,
        [tick for _, tick in _EXPECTED_JP_BANDS[1:]] + [_EXPECTED_JP_TOP],
    )
]


class TestTickSize:
    @pytest.mark.parametrize(("price", "expected"), _BOUNDARY_CASES)
    def test_jp_exact_boundary_uses_lower_band(self, price: int, expected: int) -> None:
        assert tick_size("jp", float(price)) == float(expected)

    @pytest.mark.parametrize(("price", "expected"), _ABOVE_BOUNDARY_CASES)
    def test_jp_one_above_boundary_uses_next_band(self, price: int, expected: int) -> None:
        assert tick_size("jp", float(price)) == float(expected)

    def test_above_boundary_cases_cover_expected_values(self) -> None:
        # 期待値の組み立てが手で書いた値と合うか（3,001 円は5円、最上の帯は 100,000 円）
        assert _ABOVE_BOUNDARY_CASES[0] == (3_001, 5)
        assert _ABOVE_BOUNDARY_CASES[4] == (300_001, 500)
        assert _ABOVE_BOUNDARY_CASES[-1] == (50_000_001, 100_000)

    @pytest.mark.parametrize("price", [50_000_001.0, 100_000_000.0, 1_000_000_000.0])
    def test_jp_top_band(self, price: float) -> None:
        assert tick_size("jp", price) == 100_000.0

    @pytest.mark.parametrize("price", [0.5, 1.0, 2_999.5])
    def test_jp_low_prices_use_one_yen(self, price: float) -> None:
        assert tick_size("jp", price) == 1.0

    @pytest.mark.parametrize("price", [0.01, 1.0, 123.45, 3_001.0, 100_000_000.0])
    def test_us_is_one_cent_for_any_price(self, price: float) -> None:
        assert tick_size("us", price) == 0.01


class TestRoundLimitPrice:
    @pytest.mark.parametrize(("price", "_tick"), _BOUNDARY_CASES)
    @pytest.mark.parametrize("is_buy_side", [True, False])
    def test_jp_exact_boundary_is_unchanged(
        self, price: int, _tick: int, is_buy_side: bool
    ) -> None:
        # 帯の上限は単位の倍数なので、買いも売りもそのまま
        assert round_limit_price("jp", float(price), is_buy_side) == float(price)

    @pytest.mark.parametrize(
        ("price", "expected_buy", "expected_sell"),
        [
            (3_001.0, 3_000.0, 3_005.0),
            (5_001.0, 5_000.0, 5_010.0),
            (30_001.0, 30_000.0, 30_050.0),
            (50_001.0, 50_000.0, 50_100.0),
            (300_001.0, 300_000.0, 300_500.0),
            (500_001.0, 500_000.0, 501_000.0),
            (3_000_001.0, 3_000_000.0, 3_005_000.0),
            (5_000_001.0, 5_000_000.0, 5_010_000.0),
            (30_000_001.0, 30_000_000.0, 30_050_000.0),
            (50_000_001.0, 50_000_000.0, 50_100_000.0),
        ],
    )
    def test_jp_one_above_boundary(
        self, price: float, expected_buy: float, expected_sell: float
    ) -> None:
        # 買いは切り下げ、売りは切り上げ（丸める前の値段の帯の単位で丸める）
        assert round_limit_price("jp", price, True) == expected_buy
        assert round_limit_price("jp", price, False) == expected_sell

    def test_jp_top_band(self) -> None:
        assert round_limit_price("jp", 123_456_789.0, True) == 123_400_000.0
        assert round_limit_price("jp", 123_456_789.0, False) == 123_500_000.0
        assert round_limit_price("jp", 200_000_000.0, True) == 200_000_000.0

    def test_jp_mid_band(self) -> None:
        assert round_limit_price("jp", 4_003.0, True) == 4_000.0
        assert round_limit_price("jp", 4_003.0, False) == 4_005.0
        assert round_limit_price("jp", 1_234.5, True) == 1_234.0
        assert round_limit_price("jp", 1_234.5, False) == 1_235.0

    def test_us_rounds_to_cent(self) -> None:
        assert round_limit_price("us", 123.456, True) == pytest.approx(123.45)
        assert round_limit_price("us", 123.456, False) == pytest.approx(123.46)
        assert round_limit_price("us", 123.45, True) == pytest.approx(123.45)
        assert round_limit_price("us", 123.45, False) == pytest.approx(123.45)

    @pytest.mark.parametrize("market", ["jp", "us"])
    @pytest.mark.parametrize("price", [0.0, -0.0, -1.0, -3_001.0])
    @pytest.mark.parametrize("is_buy_side", [True, False])
    def test_zero_or_negative_returns_zero(
        self, market: str, price: float, is_buy_side: bool
    ) -> None:
        assert round_limit_price(market, price, is_buy_side) == 0.0

    def test_buy_floored_to_zero_returns_min_tick_jp(self) -> None:
        assert round_limit_price("jp", 0.5, True) == 1.0

    def test_buy_floored_to_zero_returns_min_tick_us(self) -> None:
        assert round_limit_price("us", 0.005, True) == pytest.approx(0.01)

    def test_sell_small_positive_rounds_up_to_min_tick(self) -> None:
        assert round_limit_price("jp", 0.5, False) == 1.0
        assert round_limit_price("us", 0.005, False) == pytest.approx(0.01)
