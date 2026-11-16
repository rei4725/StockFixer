"""評価額の流れの現金は、SELL を持ち分までで切る（#1899）。

約定のたどり直し（replay_paper_cash）・brokers/paper/fill.py と同じ決まりにそろえる。
"""

import pandas as pd

from src.trading.paper_cash_events import replay_paper_cash
from src.trading.paper_equity import _build_equity_parts, build_equity_series

_INITIAL = 1_000_000.0


def _orders(rows):
    return pd.DataFrame(
        rows, columns=["market", "symbol", "side", "qty", "fill_price", "filled_at"]
    )


def _flat_loader(price=100.0):
    def loader(market, symbol):
        return pd.Series(price, index=pd.bdate_range("2025-12-01", periods=60))

    return loader


def test_oversell_cash_matches_replay():
    """持ち分を超える SELL を含む約定で、最後の日の現金が replay_paper_cash と一致する"""
    idx = pd.bdate_range("2026-01-05", periods=10)
    orders = _orders(
        [
            ("jp", "7203", 1, 100, 100.0, idx[1]),
            ("jp", "7203", 2, 300, 110.0, idx[3]),
            ("jp", "6758", 2, 50, 200.0, idx[4]),
        ]
    )

    equity, gross, _, _ = _build_equity_parts(orders, _flat_loader(), _INITIAL, idx)

    assert equity.iloc[-1] == replay_paper_cash(orders, _INITIAL)
    # 保有が残らないため、最後の日の評価額は現金そのもの
    assert gross.iloc[-1] == 0.0
    assert equity.iloc[-1] == _INITIAL - 10_000.0 + 11_000.0


def test_oversell_after_partial_holdings_caps_at_held_qty():
    """2回に分けた売りの2回目が持ち分を超えるとき、残りの持ち分までだけ現金に足す"""
    idx = pd.bdate_range("2026-01-05", periods=10)
    orders = _orders(
        [
            ("jp", "7203", 1, 100, 100.0, idx[0]),
            ("jp", "7203", 2, 60, 120.0, idx[2]),
            ("jp", "7203", 2, 100, 130.0, idx[4]),
        ]
    )

    equity, gross, _, _ = _build_equity_parts(orders, _flat_loader(), _INITIAL, idx)

    assert equity.iloc[-1] == replay_paper_cash(orders, _INITIAL)
    assert gross.iloc[-1] == 0.0
    assert equity.iloc[-1] == _INITIAL - 10_000.0 + 60 * 120.0 + 40 * 130.0


def test_within_holdings_equity_unchanged():
    """持ち分を超えない約定だけのときは、今までと同じ評価額になる"""
    idx = pd.bdate_range("2026-01-05", periods=10)
    orders = _orders(
        [
            ("jp", "7203", 1, 100, 100.0, idx[0]),
            ("jp", "7203", 2, 40, 120.0, idx[2]),
            ("jp", "6758", 1, 10, 500.0, idx[3]),
        ]
    )

    equity = build_equity_series(orders, _flat_loader(110.0), _INITIAL, idx)

    # 初日: 現金 −10,000、100株×110 → +1,000
    assert equity.iloc[0] == _INITIAL + 1_000.0
    # 売りの後: 現金 −10,000 + 4,800、60株×110 = 6,600 → +1,400
    assert equity.iloc[2] == _INITIAL + 1_400.0
    # 6758 を 500 で買い 110 で評価: −5,000 + 1,100 → +1,400 − 3,900 = −2,500
    assert equity.iloc[-1] == _INITIAL - 2_500.0


def test_oversell_after_split_caps_at_split_adjusted_holdings():
    """分割前に 100 株を買い、2分割の後に 300 株を売ると、直した持ち分の 200 株までで切る"""
    idx = pd.bdate_range("2026-01-05", periods=10)
    orders = _orders(
        [
            ("jp", "7203", 1, 100, 100.0, idx[0]),
            ("jp", "7203", 2, 300, 55.0, idx[7]),
        ]
    )

    def splits(market, symbol):
        return pd.Series([2.0], index=[idx[5]])

    equity, gross, _, _ = _build_equity_parts(
        orders, _flat_loader(50.0), _INITIAL, idx, split_loader=splits
    )

    assert gross.iloc[-1] == 0.0
    assert equity.iloc[-1] == _INITIAL - 10_000.0 + 200 * 55.0
