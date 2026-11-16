"""rule_best_history の売りだけの印（#1889）。

売りだけの行（保有中だからと残した行・#1801 の行）を、週の rule_best_diff と
日ごとの backtest_gap で、その週に選ばれたルールとして数えない。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.scoring import db as sdb
from src.scoring.rule_report import _backtest_gap, _rule_best_diff
from src.utils.db._connection import _db_connection
from src.utils.db.rule_results import mark_rule_best_sell_only, upsert_rule_best

D = date(2026, 11, 15)


def _best(symbol: str) -> None:
    upsert_rule_best(
        market="jp",
        symbol=symbol,
        best_rule="rsi",
        win_rate=0.6,
        net_profit=1000.0,
        num_trades=10,
        profit_factor=1.5,
        max_drawdown=-0.2,
        backtest_start="2024-01-01",
        backtest_end="2024-06-30",
    )


def test_snapshot_copies_sell_only() -> None:
    _best("SOA")
    _best("SOB")
    mark_rule_best_sell_only("jp", "SOB")
    sdb.snapshot_rule_best(D, "jp")
    with _db_connection() as con:
        rows = con.execute(
            "SELECT symbol, sell_only FROM rule_best_history"
            " WHERE evaluated_on = %s AND market = %s AND symbol IN ('SOA', 'SOB')"
            " ORDER BY symbol",
            [D, "jp"],
        ).fetchall()
    assert [tuple(r) for r in rows] == [("SOA", False), ("SOB", True)]
    hist = sdb.load_rule_best_history("jp")
    got = hist[hist["symbol"].isin(["SOA", "SOB"])].set_index("symbol")["sell_only"]
    assert got.to_dict() == {"SOA": False, "SOB": True}


def _hist(rows: list[tuple[date, str, str, float, object]]) -> pd.DataFrame:
    cols = ["evaluated_on", "market", "symbol", "best_rule", "win_rate", "num_trades", "sell_only"]
    return pd.DataFrame([(d, "jp", s, r, w, 10, so) for d, s, r, w, so in rows], columns=cols)


def test_rule_best_diff_sell_only_is_removed_not_added() -> None:
    hist = _hist(
        [
            (date(2026, 11, 1), "A", "rsi", 0.6, False),
            (date(2026, 11, 1), "B", "macd", 0.6, False),
            (date(2026, 11, 8), "A", "rsi", 0.6, True),
            (date(2026, 11, 8), "B", "macd", 0.6, False),
            (date(2026, 11, 8), "C", "bollinger", 0.4, True),
        ]
    )
    diff = _rule_best_diff(hist, date(2026, 11, 15))
    assert diff["evaluated_on"] == "2026-11-08"
    assert diff["previous_evaluated_on"] == "2026-11-01"
    # A は前の週に選ばれ、今週は売りだけ。C は売りだけの行なので選ばれたルールでない
    assert diff["removed"] == [{"symbol": "A", "rule": "rsi"}]
    assert diff["added"] == []
    assert diff["changed"] == []


def test_rule_best_diff_old_rows_without_column() -> None:
    hist = _hist(
        [
            (date(2026, 11, 1), "A", "rsi", 0.6, None),
            (date(2026, 11, 8), "B", "rsi", 0.6, None),
        ]
    ).drop(columns=["sell_only"])
    diff = _rule_best_diff(hist, date(2026, 11, 15))
    assert diff["added"] == [{"symbol": "B", "rule": "rsi"}]
    assert diff["removed"] == [{"symbol": "A", "rule": "rsi"}]
    # 列はあっても値の無い行（NULL）は選ばれたルールとして読む
    hist2 = _hist(
        [
            (date(2026, 11, 1), "A", "rsi", 0.6, None),
            (date(2026, 11, 8), "B", "rsi", 0.6, None),
        ]
    )
    assert _rule_best_diff(hist2, date(2026, 11, 15)) == diff


def _scored() -> pd.DataFrame:
    """A の brk の BUY が 10-01・10-05・10-12 に1件ずつ（どれも当たり）。"""
    days = [date(2026, 10, 1), date(2026, 10, 5), date(2026, 10, 12)]
    return pd.DataFrame(
        {
            "date": days,
            "symbol": ["A"] * 3,
            "key": ["brk"] * 3,
            "direction": [1] * 3,
            "horizon": [5] * 3,
            "hit": [True] * 3,
            "baseline": [0.5] * 3,
            "signed_return": [0.01] * 3,
        }
    )


def test_backtest_gap_skips_sell_only_evaluation() -> None:
    # 10-04 の評価で A は売りだけになり、10-11 に選ばれ直す
    hist = _hist(
        [
            (date(2026, 9, 1), "A", "brk", 0.7, False),
            (date(2026, 10, 4), "A", "brk", 0.3, True),
            (date(2026, 10, 11), "A", "brk", 0.8, False),
        ]
    )
    by_side, combined = _backtest_gap(_scored(), hist, pd.DataFrame())
    buy = by_side["brk"]["buy"]
    # 10-05 の BUY は売りだけの評価に当たるので n_h5 に入れず、別の欄で数える
    assert buy["n_h5"] == 2
    assert buy["n_sell_only"] == 1
    assert buy["n_unmatched"] == 0
    assert abs(buy["backtest_win_rate"] - 0.75) < 1e-9
    assert by_side["brk"]["sell"]["n_sell_only"] == 0
    assert combined["brk"]["n_h5"] == 2
    assert combined["brk"]["n_sell_only"] == 1


def test_backtest_gap_sell_only_fallback_to_now() -> None:
    # 歴史が無いときの今の行が売りだけなら、近似でも数えない
    now = pd.DataFrame(
        {
            "market": ["jp"],
            "symbol": ["A"],
            "best_rule": ["brk"],
            "win_rate": [0.3],
            "num_trades": [10],
            "sell_only": [True],
        }
    )
    _, combined = _backtest_gap(_scored(), pd.DataFrame(), now)
    assert combined["brk"]["n_h5"] == 0
    assert combined["brk"]["n_sell_only"] == 3


def test_backtest_gap_old_rows_without_column() -> None:
    hist = _hist([(date(2026, 9, 1), "A", "brk", 0.7, None)]).drop(columns=["sell_only"])
    _, combined = _backtest_gap(_scored(), hist, pd.DataFrame())
    assert combined["brk"]["n_h5"] == 3
    assert combined["brk"]["n_sell_only"] == 0
    assert abs(combined["brk"]["backtest_win_rate"] - 0.7) < 1e-9
