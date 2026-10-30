"""指値の値段を呼値の単位に丸める（#1396）。"""

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

# 東証の一般の呼値の単位（TOPIX500 以外の銘柄の表。TOPIX500 の細かい単位の倍数になる）。
# (この値段以下, 単位)。最後の帯より上は _JP_TICK_TOP を使う
_JP_TICK_TABLE: tuple[tuple[int, int], ...] = (
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
_JP_TICK_TOP = 100_000
_US_TICK = Decimal("0.01")


def _jp_tick(price: Decimal) -> Decimal:
    for upper, tick in _JP_TICK_TABLE:
        if price <= upper:
            return Decimal(tick)
    return Decimal(_JP_TICK_TOP)


def tick_size(market: str, price: float) -> float:
    """その値段の帯の呼値の単位を返す（#1403）。"""
    if market != "jp":
        return float(_US_TICK)
    return float(_jp_tick(Decimal(str(price))))


def round_limit_price(market: str, price: float, is_buy_side: bool) -> float:
    """指値を呼値の単位に丸める。買いは切り下げ、売りは切り上げる。

    丸める前の値段の帯の単位で丸める。帯の上限は単位の倍数なので、丸めた値も
    その帯の中（か、より細かい単位の帯の倍数）に収まる。
    買いを切り下げて 0 以下になるときは、最小の単位を返す。
    """
    if price <= 0:
        return 0.0
    value = Decimal(str(price))
    tick = _jp_tick(value) if market == "jp" else _US_TICK
    rounding = ROUND_FLOOR if is_buy_side else ROUND_CEILING
    rounded = (value / tick).to_integral_value(rounding=rounding) * tick
    if rounded <= 0:
        rounded = tick
    return float(rounded)
