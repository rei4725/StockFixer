"""紙の注文から FIFO で往復を組み、道別・月別にまとめる関数のテスト（#1817）"""

from datetime import datetime

import pandas as pd
import pytest

from src.reporting.paper_roundtrips import (
    ROUTE_PREDICTION,
    ROUTE_RULE,
    ROUTE_UNKNOWN,
    build_roundtrips,
    classify_route,
    count_phantom_sells,
    load_paper_roundtrip_orders,
    summarize_roundtrips,
)


def _orders() -> pd.DataFrame:
    rows = [
        # 予想の道（23:50 UTC）で 100 株買い、2回に分けて売る（一部約定）
        ("o1", "A", 1, 100, 1000.0, datetime(2026, 9, 1, 23, 50), datetime(2026, 9, 2, 0, 0)),
        ("o2", "A", 2, 40, 1100.0, datetime(2026, 9, 3, 7, 30), datetime(2026, 9, 4, 0, 0)),
        ("o3", "A", 2, 60, 900.0, datetime(2026, 10, 1, 7, 30), datetime(2026, 10, 2, 0, 0)),
        # ルールの道（07:30 UTC）で 2 回買い、1 回の売りで FIFO に両方を閉じる
        ("o4", "B", 1, 10, 500.0, datetime(2026, 9, 5, 7, 30), datetime(2026, 9, 6, 0, 0)),
        ("o5", "B", 1, 10, 600.0, datetime(2026, 9, 7, 7, 30), datetime(2026, 9, 8, 0, 0)),
        ("o6", "B", 2, 15, 700.0, datetime(2026, 9, 9, 7, 30), datetime(2026, 9, 10, 0, 0)),
        # 時刻で見分けられない買い。対になる売りは無い
        ("o7", "C", 1, 5, 300.0, datetime(2026, 9, 11, 21, 0), datetime(2026, 9, 12, 0, 0)),
    ]
    return pd.DataFrame(
        rows,
        columns=["order_id", "symbol", "side", "qty", "fill_price", "created_at", "filled_at"],
    )


def test_classify_route_by_utc_hour():
    assert classify_route(datetime(2026, 9, 1, 23, 50)) == ROUTE_PREDICTION
    assert classify_route(datetime(2026, 9, 1, 7, 31)) == ROUTE_RULE
    assert classify_route(datetime(2026, 9, 1, 21, 0)) == ROUTE_UNKNOWN


def test_build_roundtrips_fifo_with_partial_and_open_buy():
    trips, open_lots, _ = build_roundtrips(_orders())
    got = sorted((t.symbol, t.qty, t.route, round(t.pnl, 2), t.exit_month) for t in trips)
    assert got == [
        ("A", 40, ROUTE_PREDICTION, 4000.0, "2026-09"),
        ("A", 60, ROUTE_PREDICTION, -6000.0, "2026-10"),
        ("B", 5, ROUTE_RULE, 500.0, "2026-09"),
        ("B", 10, ROUTE_RULE, 2000.0, "2026-09"),
    ]
    assert trips[0].holding_days == 2
    # B の 2 つ目の買いの残り 5 株と、C の 5 株が開いたまま
    assert sorted((lot.symbol, lot.qty) for lot in open_lots) == [("B", 5), ("C", 5)]


def test_sell_beyond_open_lots_is_counted_as_unmatched():
    df = _orders()
    extra = pd.DataFrame(
        [("o8", "D", 2, 7, 100.0, datetime(2026, 9, 1, 7, 30), datetime(2026, 9, 2, 0, 0))],
        columns=df.columns,
    )
    trips, _, unmatched = build_roundtrips(pd.concat([df, extra], ignore_index=True))
    assert all(t.symbol != "D" for t in trips)
    assert unmatched == {"D": 7}


def test_sell_without_realized_pnl_does_not_consume_lots():
    # 本番の 8766 のように、同じ時刻に約定した売りが2本あり、片方は保有が無く
    # realized_pnl が空のまま filled になった。その売りは買いを食わない
    rows = [
        ("b1", "E", 1, 100, 500.0, datetime(2026, 10, 1, 23, 51), datetime(2026, 10, 2), None),
        ("s0", "E", 2, 100, 515.0, datetime(2026, 10, 2, 7, 30), datetime(2026, 10, 5), None),
        ("s1", "E", 2, 100, 515.0, datetime(2026, 10, 4, 23, 51), datetime(2026, 10, 5), 1500.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "order_id",
            "symbol",
            "side",
            "qty",
            "fill_price",
            "created_at",
            "filled_at",
            "realized_pnl",
        ],
    )
    trips, open_lots, unmatched = build_roundtrips(df)
    assert [(t.qty, t.pnl, t.route) for t in trips] == [(100, 1500.0, ROUTE_PREDICTION)]
    assert open_lots == [] and unmatched == {}
    assert count_phantom_sells(df) == 1
    assert count_phantom_sells(_orders()) == 0


def test_summarize_by_route_and_month():
    trips, _, _ = build_roundtrips(_orders())
    summary = summarize_roundtrips(trips)
    by_route = {r.key: r for r in summary.by_route}
    assert by_route[ROUTE_PREDICTION].trips == 2
    assert by_route[ROUTE_PREDICTION].wins == 1
    assert by_route[ROUTE_PREDICTION].pnl == pytest.approx(-2000.0)
    assert by_route[ROUTE_RULE].trips == 2
    assert by_route[ROUTE_RULE].win_rate == pytest.approx(1.0)
    assert by_route[ROUTE_RULE].pnl == pytest.approx(2500.0)
    by_month = {(r.route, r.key): r.pnl for r in summary.by_month}
    assert by_month[(ROUTE_PREDICTION, "2026-10")] == pytest.approx(-6000.0)
    assert by_month[(ROUTE_RULE, "2026-09")] == pytest.approx(2500.0)
    assert summary.total_pnl == pytest.approx(500.0)


def test_load_paper_roundtrip_orders_reads_filled_buys_and_sells():
    class _Cur:
        description = [("order_id",), ("symbol",)]

        def fetchall(self):
            return [("o1", "A")]

    class _Con:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            self.calls.append((sql, params))
            return _Cur()

    con = _Con()
    df = load_paper_roundtrip_orders(con)
    assert list(df.columns) == ["order_id", "symbol"] and len(df) == 1
    sql, params = con.calls[0]
    assert "status = 'filled'" in sql and "realized_pnl" in sql
    assert params == [1, 2]


_SPLIT_COLUMNS = [
    "order_id",
    "market",
    "symbol",
    "side",
    "qty",
    "fill_price",
    "created_at",
    "filled_at",
]


def _split_orders() -> pd.DataFrame:
    # 分割の前に 100 株を 1,000 円で買い、2分割（2026-09-10）の後に 200 株を 500 円で売る
    rows = [
        ("b1", "jp", "S", 1, 100, 1000.0, datetime(2026, 9, 1, 23, 50), datetime(2026, 9, 2)),
        ("s1", "jp", "S", 2, 200, 500.0, datetime(2026, 9, 15, 7, 30), datetime(2026, 9, 16)),
    ]
    return pd.DataFrame(rows, columns=_SPLIT_COLUMNS)


def _splits(market, symbol):
    if symbol != "S":
        return None
    return pd.Series([2.0], index=pd.DatetimeIndex([pd.Timestamp("2026-09-10")]))


def test_split_between_buy_and_sell_is_adjusted_to_current_level():
    calls = []

    def loader(market, symbol):
        calls.append((market, symbol))
        return _splits(market, symbol)

    trips, open_lots, unmatched = build_roundtrips(_split_orders(), split_loader=loader)
    assert [(t.qty, t.entry_price, t.exit_price, t.pnl) for t in trips] == [
        (200, 500.0, 500.0, 0.0)
    ]
    assert open_lots == [] and unmatched == {}
    assert calls == [("jp", "S")]


def test_split_unknown_symbol_keeps_raw_values():
    # 分割の記録が取れない銘柄は生の値で組む（今までと同じ）
    trips, _, unmatched = build_roundtrips(_split_orders(), split_loader=lambda m, s: None)
    assert [(t.qty, t.pnl) for t in trips] == [(100, -50000.0)]
    assert unmatched == {"S": 100}


def test_split_loader_does_not_change_orders_without_splits():
    df = _orders()
    df.insert(1, "market", "jp")
    base = build_roundtrips(df)
    with_loader = build_roundtrips(df, split_loader=_splits)
    assert with_loader == base


def test_empty_orders():
    trips, open_lots, _ = build_roundtrips(pd.DataFrame(columns=_orders().columns))
    assert trips == [] and open_lots == []
    assert summarize_roundtrips(trips).total_pnl == 0.0
