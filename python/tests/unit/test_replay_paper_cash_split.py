"""約定のたどり直し（replay_paper_cash）は、持ち分を分割の比で直した株数で数える（#1912）。

評価額の流れ（#1899）・paper_broker（#899）と同じ決まりにそろえる。
"""

import pandas as pd

from src.trading.paper_cash_events import replay_paper_cash
from src.trading.paper_equity import _build_equity_parts

_INITIAL = 1_000_000.0


def _orders(rows):
    return pd.DataFrame(
        rows, columns=["market", "symbol", "side", "qty", "fill_price", "filled_at"]
    )


def _flat_loader(price=100.0):
    def loader(market, symbol):
        return pd.Series(price, index=pd.bdate_range("2025-12-01", periods=60))

    return loader


def test_sell_after_split_matches_equity_cash():
    """分割の前に 100 株買い、2分割の後に 200 株売ると、現金が評価額の流れの最後の日と一致する"""
    idx = pd.bdate_range("2026-01-05", periods=10)
    orders = _orders(
        [
            ("jp", "7203", 1, 100, 100.0, idx[0]),
            ("jp", "7203", 2, 200, 55.0, idx[7]),
        ]
    )

    def splits(market, symbol):
        return pd.Series([2.0], index=[idx[5]])

    equity, gross, _, _ = _build_equity_parts(
        orders, _flat_loader(50.0), _INITIAL, idx, split_loader=splits
    )
    cash = replay_paper_cash(orders, _INITIAL, split_loader=splits)

    assert gross.iloc[-1] == 0.0
    assert cash == equity.iloc[-1]
    assert cash == _INITIAL - 10_000.0 + 200 * 55.0


def test_oversell_after_split_caps_at_split_adjusted_holdings():
    """2分割の後に 300 株売ると、直した持ち分の 200 株までで切る"""
    idx = pd.bdate_range("2026-01-05", periods=10)
    orders = _orders(
        [
            ("jp", "7203", 1, 100, 100.0, idx[0]),
            ("jp", "7203", 2, 300, 55.0, idx[7]),
        ]
    )

    def splits(market, symbol):
        return pd.Series([2.0], index=[idx[5]])

    assert replay_paper_cash(orders, _INITIAL, split_loader=splits) == (
        _INITIAL - 10_000.0 + 200 * 55.0
    )


def test_no_split_records_keeps_raw_qty():
    """分割の記録が取れない銘柄（None）は、今までどおり生の株数で数える"""
    idx = pd.bdate_range("2026-01-05", periods=10)
    orders = _orders(
        [
            ("jp", "7203", 1, 100, 100.0, idx[0]),
            ("jp", "7203", 2, 200, 55.0, idx[7]),
        ]
    )

    cash = replay_paper_cash(orders, _INITIAL, split_loader=lambda market, symbol: None)

    assert cash == _INITIAL - 10_000.0 + 100 * 55.0
    assert cash == replay_paper_cash(orders, _INITIAL)


def test_without_split_cash_unchanged():
    """分割の無い約定では、split_loader を渡しても渡さなくても今までと同じ現金になる"""
    idx = pd.bdate_range("2026-01-05", periods=10)
    orders = _orders(
        [
            ("jp", "7203", 1, 100, 100.0, idx[0]),
            ("jp", "7203", 2, 60, 120.0, idx[2]),
            ("jp", "7203", 2, 100, 130.0, idx[4]),
            ("us", "AAPL", 3, 10, 200.0, idx[1]),
            ("us", "AAPL", 4, 10, 180.0, idx[3]),
        ]
    )
    expected = _INITIAL - 10_000.0 + 60 * 120.0 + 40 * 130.0 + (200.0 - 180.0) * 10

    def no_splits(market, symbol):
        return pd.Series(dtype=float)

    assert replay_paper_cash(orders, _INITIAL) == expected
    assert replay_paper_cash(orders, _INITIAL, split_loader=no_splits) == expected
