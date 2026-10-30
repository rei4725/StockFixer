"""ユニットテスト: 分割の直しが、読んだ後に約定で変わった保有の株数を消さない（#1423）

分割の記録を取っている間（読んだ後、書く前）に、同じ銘柄の約定が分割を当てて確定した形を、
get_splits の中で行を書き換えて再現する。実 Postgres（conftest の `_isolate_db` でロールバック分離）。
"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.trading.brokers.paper import split_adjust as sa
from src.utils.db._connection import _db_connection

# 1:2 の分割の権利落ちの日 D（2026-03-10）の 9:05 JST（tz 無しの UTC）
_D_0905_JST = datetime(2026, 3, 10, 0, 5)
_BEFORE_SPLIT = "2026-03-02 01:00:00"


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(sa, "_now_utc", lambda: datetime(2026, 3, 10, 0, 10))
    with _db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute("DELETE FROM paper_short_positions")


def _market_data_with_fill_between(fill_sql: str, params: list) -> MagicMock:
    """分割の記録を返す前に、別の約定が行を書き換える"""
    index = pd.DatetimeIndex(["2026-03-10"]).tz_localize("Asia/Tokyo")
    splits = pd.Series([2.0], index=index, name="Stock Splits")

    def _get_splits(_ticker):
        with _db_connection() as con:
            con.execute(fill_sql, params)
        return splits

    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_splits.side_effect = _get_splits
    return market_data


def _long_row() -> tuple:
    with _db_connection() as con:
        return con.execute(
            "SELECT qty, avg_price FROM paper_positions WHERE symbol='7203'"
        ).fetchone()


def _short_row() -> tuple:
    with _db_connection() as con:
        return con.execute(
            "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol='7203'"
        ).fetchone()


def _insert_long() -> None:
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
            "VALUES ('7203', 100, 2000.0, %s)",
            [_BEFORE_SPLIT],
        )


def test_buy_filled_between_read_and_write_keeps_bought_shares():
    _insert_long()
    # 約定が保有を 200株 @ 1000 に直し、100株 @ 1000 を足して確定した
    market_data = _market_data_with_fill_between(
        "UPDATE paper_positions SET qty=300, avg_price=1000.0, updated_at=%s WHERE symbol='7203'",
        [_D_0905_JST],
    )
    qty, avg, known = sa.adjust_position_for_splits(market_data, "7203", 100, 2000.0, _BEFORE_SPLIT)
    assert known
    assert _long_row() == (300, pytest.approx(1000.0))
    assert (qty, avg) == (300, pytest.approx(1000.0))


def test_sell_filled_between_read_and_write_keeps_sold_shares_out():
    _insert_long()
    # 約定が保有を 200株 に直し、100株 を売って確定した
    market_data = _market_data_with_fill_between(
        "UPDATE paper_positions SET qty=100, avg_price=1000.0, updated_at=%s WHERE symbol='7203'",
        [_D_0905_JST],
    )
    qty, _, _ = sa.adjust_position_for_splits(market_data, "7203", 100, 2000.0, _BEFORE_SPLIT)
    assert _long_row() == (100, pytest.approx(1000.0))
    assert qty == 100


def test_sell_of_unadjusted_row_between_read_and_write_applies_split_to_rest():
    _insert_long()
    # 分割を当てない道で 40株 だけ売れて、updated_at は分割の前のまま残った行
    market_data = _market_data_with_fill_between(
        "UPDATE paper_positions SET qty=60 WHERE symbol='7203'", []
    )
    qty, avg, _ = sa.adjust_position_for_splits(market_data, "7203", 100, 2000.0, _BEFORE_SPLIT)
    assert _long_row() == (120, pytest.approx(1000.0))
    assert (qty, avg) == (120, pytest.approx(1000.0))


def test_short_cover_filled_between_read_and_write_keeps_covered_shares_out():
    with _db_connection() as con:
        con.execute(
            "INSERT INTO paper_short_positions "
            "(symbol, qty, avg_short_price, opened_at, updated_at) "
            "VALUES ('7203', 100, 2000.0, %s, %s)",
            [_BEFORE_SPLIT, _BEFORE_SPLIT],
        )
    # 約定が空売りを 200株 に直し、150株 を返済して確定した
    market_data = _market_data_with_fill_between(
        "UPDATE paper_short_positions SET qty=50, avg_short_price=1000.0, updated_at=%s "
        "WHERE symbol='7203'",
        [_D_0905_JST],
    )
    qty, _, _ = sa.adjust_short_position_for_splits(market_data, "7203", 100, 2000.0, _BEFORE_SPLIT)
    assert _short_row() == (50, pytest.approx(1000.0))
    assert qty == 50


def test_position_sold_out_between_read_and_write_is_not_written_back():
    _insert_long()
    market_data = _market_data_with_fill_between(
        "DELETE FROM paper_positions WHERE symbol='7203'", []
    )
    sa.adjust_position_for_splits(market_data, "7203", 100, 2000.0, _BEFORE_SPLIT)
    assert _long_row() is None
