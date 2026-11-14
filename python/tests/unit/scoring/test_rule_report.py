"""日次と週次の採点（src.scoring.rule_report）のテスト。DB の読み込みは差し替える。"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.scoring.rule_report import (
    ScoringLoaders,
    build_rule_accuracy,
    prediction_signals,
    write_rule_accuracy,
    write_weekly_rule_accuracy,
)

AS_OF = date(2026, 11, 4)
S_KEYS = {"n", "hit_rate", "baseline_rate", "edge", "mean_signed_return"}


def _bdays(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.bdate_range(start, end))


def _bars() -> pd.DataFrame:
    """日足: A は毎日 +1%、B は毎日 -1%、C は横ばい（2026-09-01 から AS_OF まで）。"""
    rows = []
    for i, ts in enumerate(_bdays("2026-09-01", AS_OF.isoformat())):
        for sym, step in (("A", 1.01), ("B", 0.99), ("C", 1.0)):
            px = 100.0 * step**i
            rows.append((sym, ts + pd.Timedelta(hours=15), px, px))
    return pd.DataFrame(rows, columns=["symbol", "ts", "close", "adj_close"])


def _rule_signals(empty: bool = False) -> pd.DataFrame:
    cols = ["signal_date", "market", "symbol", "rule_name", "signal", "price", "app_version"]
    if empty:
        return pd.DataFrame(columns=cols)
    rows: list[tuple[date, str, str, str, int, float, str | None]] = []
    for ts in _bdays("2026-09-01", "2026-10-30"):
        d = ts.date()
        rows.append((d, "jp", "A", "rsi", 1, 100.0, "2.37.0"))  # 当たり
        rows.append((d, "jp", "B", "bollinger", 1, 100.0, None))  # 外れ
        rows.append((d, "jp", "C", "rsi", -1, 100.0, None))  # 0 の動きは外れ
    return pd.DataFrame(rows, columns=cols)


def _h1() -> pd.DataFrame:
    rows = []
    for ts in _bdays("2026-10-01", "2026-10-28"):
        pa = (ts + pd.Timedelta(hours=22, minutes=40)).strftime("%Y%m%d_%H%M%S")
        rows.append(("A", "production", pa, 0.01))
        rows.append(("B", "challenger", pa, -0.01))
    return pd.DataFrame(rows, columns=["symbol", "model_name", "predicted_at", "predicted_ratio"])


def _history(empty: bool = False) -> pd.DataFrame:
    cols = [
        "market",
        "symbol",
        "predicted_at",
        "model_version",
        "current_price",
        "diff_ratio",
        "diff_ratio_5d",
        "app_version",
        "logged_at",
    ]
    if empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for ts in _bdays("2026-10-01", "2026-10-20"):
        pa = (ts + pd.Timedelta(hours=22, minutes=40)).strftime("%Y%m%d_%H%M%S")
        rows.append(("jp", "A", pa, "production", 100.0, 0.01, 0.03, "2.37.0", ts))
    return pd.DataFrame(rows, columns=cols)


def _best() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": ["jp", "jp", "jp"],
            "symbol": ["A", "B", "C"],
            "best_rule": ["rsi", "bollinger", "rsi"],
            "win_rate": [0.7, 0.6, 0.5],
            "num_trades": [10, 10, 10],
        }
    )


def _best_history(rows: list[tuple[date, str, str, float]] | None = None) -> pd.DataFrame:
    cols = ["evaluated_on", "market", "symbol", "best_rule", "win_rate", "num_trades"]
    rows = rows or []
    return pd.DataFrame([(d, "jp", s, r, w, 10) for d, s, r, w in rows], columns=cols)


def _loaders(
    *,
    empty_signals: bool = False,
    empty_history: bool = False,
    best_history: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
) -> ScoringLoaders:
    return ScoringLoaders(
        bars=lambda market, start: _bars(),
        rule_signals=lambda market, start: _rule_signals(empty_signals),
        prediction_h1=lambda market, start: _h1(),
        prediction_history=lambda market, start: _history(empty_history),
        rule_best=lambda market: _best(),
        rule_best_history=lambda market: (
            best_history if best_history is not None else _best_history()
        ),
        rule_outcomes_scored=lambda market, start, end: (
            outcomes if outcomes is not None else pd.DataFrame()
        ),
    )


def test_report_shape_and_keys() -> None:
    report, rows = build_rule_accuracy("jp", AS_OF, loaders=_loaders())
    for key in (
        "date",
        "generated_at",
        "market",
        "anchor_horizon",
        "baseline",
        "n_excluded_split",
        "windows",
        "backtest_gap",
        "backtest_gap_combined",
        "top_symbols",
    ):
        assert key in report
    assert report["anchor_horizon"] == 5
    assert report["baseline"] == "per_day_same_direction"
    assert set(report["windows"]) == {"28d", "all"}
    s = report["windows"]["28d"]["rules"]["rsi"]["buy"]["5"]
    assert set(s) == S_KEYS
    assert s["n"] > 0 and s["hit_rate"] == 1.0
    # A の買いは毎日当たり、その日の上がった割合は 1/3
    assert abs(s["baseline_rate"] - 1 / 3) < 1e-9
    assert abs(s["edge"] - 2 / 3) < 1e-9
    assert set(report["windows"]["all"]["rules"]["rsi"]) == {"buy", "sell"}
    assert set(report["windows"]["all"]["rules"]["rsi"]["buy"]) == {"1", "5", "20"}
    # C の売りは横ばいで外れ
    assert report["windows"]["all"]["rules"]["rsi"]["sell"]["5"]["hit_rate"] == 0.0
    # 28d の窓は全期間より少ない
    n_all = report["windows"]["all"]["overall"]["buy"]["5"]["n"]
    assert 0 < report["windows"]["28d"]["overall"]["buy"]["5"]["n"] < n_all
    preds = report["windows"]["all"]["predictions"]
    assert set(preds) == {"production", "challenger"}
    assert set(preds["production"]) == {"h1_claim", "h5_claim"}
    assert preds["production"]["h1_claim"]["1"]["hit_rate"] == 1.0
    assert preds["challenger"]["h1_claim"]["1"]["hit_rate"] == 1.0
    top = report["top_symbols"]
    assert 0 < len(top) <= 20
    assert set(top[0]) == {"symbol", "rule_name", "direction", "n_h5", "hit_rate_h5", "edge_h5"}
    assert [t["n_h5"] for t in top] == sorted((t["n_h5"] for t in top), reverse=True)
    # DB に書く行: シグナル×ホライズン、scored_at は必ず入る
    assert set(rows["horizon"]) == {1, 5, 20}
    assert rows["scored_at"].notna().all()
    assert set(rows.loc[rows["symbol"] == "A", "app_version"]) == {"2.37.0"}
    assert rows.loc[rows["symbol"] == "B", "app_version"].isna().all()
    assert set(rows["market"]) == {"jp"}


def test_empty_signals_writes_zero(tmp_path: Path) -> None:
    report, rows = build_rule_accuracy("jp", AS_OF, loaders=_loaders(empty_signals=True))
    s = report["windows"]["28d"]["overall"]["buy"]["5"]
    assert s["n"] == 0
    assert s["hit_rate"] is None
    assert s["edge"] is None
    assert report["windows"]["all"]["rules"] == {}
    assert report["top_symbols"] == []
    assert rows.empty
    path = write_rule_accuracy(report, AS_OF, output_dir=tmp_path)
    assert path == tmp_path / "2026-11-04.json"
    assert json.loads(path.read_text(encoding="utf-8"))["windows"]["all"]["rules"] == {}


def test_prediction_date_is_prev_bar() -> None:
    px = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "date": [date(2026, 11, 2), date(2026, 11, 4), date(2026, 11, 5)],
            "px": [100.0, 101.0, 102.0],
        }
    )
    h1 = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "model_name": ["production", "challenger"],
            # UTC 2026-11-04 22:40 = JST 2026-11-05 07:40
            "predicted_at": ["20261104_224000", "2026-11-04T22:40:00Z"],
            "predicted_ratio": [0.02, -0.01],
        }
    )
    out = prediction_signals(h1, pd.DataFrame(), px)
    assert list(out.columns) == ["date", "symbol", "key", "direction", "horizon_of_claim"]
    assert list(out["date"]) == [date(2026, 11, 4), date(2026, 11, 4)]
    by_key = dict(zip(out["key"], out["direction"]))
    assert by_key == {"production": 1, "challenger": -1}
    assert set(out["horizon_of_claim"]) == {1}


def test_prediction_signals_drop_zero_and_nan() -> None:
    px = pd.DataFrame({"symbol": ["A"], "date": [date(2026, 11, 4)], "px": [100.0]})
    h1 = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "model_name": ["production", "challenger"],
            "predicted_at": ["20261105_224000", "20261105_224000"],
            "predicted_ratio": [0.0, float("nan")],
        }
    )
    hist = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "predicted_at": ["20261105_224000", "20261105_224000"],
            "model_version": ["production", "v1"],
            "diff_ratio_5d": [0.03, 0.03],
        }
    )
    out = prediction_signals(h1, hist, px)
    # 0 と NaN は捨て、production・challenger 以外の版も捨てる
    assert len(out) == 1
    assert out.iloc[0]["horizon_of_claim"] == 5
    assert out.iloc[0]["key"] == "production"


def test_prediction_history_empty_h5_zero() -> None:
    report, _ = build_rule_accuracy("jp", AS_OF, loaders=_loaders(empty_history=True))
    prod = report["windows"]["all"]["predictions"]["production"]
    assert prod["h5_claim"]["5"]["n"] == 0
    assert prod["h5_claim"]["5"]["hit_rate"] is None
    assert prod["h1_claim"]["1"]["n"] > 0
    # history がある時は h5_claim に数が出る
    report2, _ = build_rule_accuracy("jp", AS_OF, loaders=_loaders())
    assert report2["windows"]["all"]["predictions"]["production"]["h5_claim"]["5"]["n"] > 0


def test_backtest_gap_approx_flag() -> None:
    report, _ = build_rule_accuracy("jp", AS_OF, loaders=_loaders())
    side = report["backtest_gap"]
    assert set(side) == {"rsi", "bollinger"}
    assert set(side["rsi"]) == {"buy", "sell"}
    assert set(side["rsi"]["buy"]) == {
        "backtest_win_rate",
        "live_hit_rate_h5",
        "n_h5",
        "n_unmatched",
        "approx",
        "baseline_rate",
        "edge",
    }
    assert side["rsi"]["buy"]["approx"] is True and side["rsi"]["sell"]["approx"] is True
    # bollinger は買いだけ。売りの側は数が無く、approx は立つ
    assert side["bollinger"]["sell"]["n_h5"] == 0
    assert side["bollinger"]["sell"]["approx"] is True
    gap = report["backtest_gap_combined"]
    assert set(gap) == {"rsi", "bollinger"}
    assert gap["rsi"]["approx"] is True
    assert set(gap["rsi"]) == {
        "backtest_win_rate",
        "live_hit_rate_h5",
        "n_h5",
        "n_unmatched",
        "approx",
    }
    assert gap["rsi"]["n_unmatched"] == 0 and gap["bollinger"]["n_unmatched"] == 0
    assert abs(gap["bollinger"]["backtest_win_rate"] - 0.6) < 1e-9
    assert gap["bollinger"]["live_hit_rate_h5"] == 0.0


def test_backtest_gap_uses_history_when_covered() -> None:
    hist = _best_history(
        [
            (date(2026, 8, 1), "B", "bollinger", 0.9),
            (date(2026, 10, 1), "B", "bollinger", 0.8),
        ]
    )
    report, _ = build_rule_accuracy("jp", AS_OF, loaders=_loaders(best_history=hist))
    gap = report["backtest_gap_combined"]["bollinger"]
    assert gap["approx"] is False
    # 10-01 より前のシグナルは 0.9、以後は 0.8 が当たるので、その間の値になる
    assert 0.8 < gap["backtest_win_rate"] < 0.9
    assert report["backtest_gap_combined"]["rsi"]["approx"] is True
    # 向きごとの当て方も同じ（bollinger は買いだけ）
    buy = report["backtest_gap"]["bollinger"]["buy"]
    assert buy["approx"] is False
    assert buy["backtest_win_rate"] == gap["backtest_win_rate"]
    assert buy["n_h5"] == gap["n_h5"]
    assert report["backtest_gap"]["rsi"]["buy"]["approx"] is True


def test_backtest_gap_counts_unmatched() -> None:
    # B の評価は 10-01 までは rsi、10-01 から bollinger。9 月の B の bollinger は数えず n_unmatched へ
    hist = _best_history(
        [
            (date(2026, 8, 1), "B", "rsi", 0.9),
            (date(2026, 10, 1), "B", "bollinger", 0.8),
        ]
    )
    report, rows = build_rule_accuracy("jp", AS_OF, loaders=_loaders(best_history=hist))
    gap = report["backtest_gap_combined"]["bollinger"]
    n_sept = len(_bdays("2026-09-01", "2026-09-30"))
    n_all = int(((rows["rule_name"] == "bollinger") & (rows["horizon"] == 5)).sum())
    assert gap["n_unmatched"] == n_sept
    assert gap["n_h5"] == n_all - n_sept
    assert abs(gap["backtest_win_rate"] - 0.8) < 1e-9
    assert report["backtest_gap_combined"]["rsi"]["n_unmatched"] == 0
    # 向きごとの n_unmatched（bollinger のシグナルは買いだけ）
    side = report["backtest_gap"]["bollinger"]
    assert side["buy"]["n_unmatched"] == n_sept
    assert side["sell"]["n_unmatched"] == 0
    assert side["buy"]["n_h5"] == n_all - n_sept
    assert abs(side["buy"]["backtest_win_rate"] - 0.8) < 1e-9
    assert report["backtest_gap"]["rsi"]["buy"]["n_unmatched"] == 0


def _side_loaders() -> ScoringLoaders:
    """brk のルールが A に買い1件（当たり）と売り3件（外れ）を出す。A の評価は brk。"""
    cols = ["signal_date", "market", "symbol", "rule_name", "signal", "price", "app_version"]
    sig = pd.DataFrame(
        [
            (date(2026, 10, 1), "jp", "A", "brk", 1, 100.0, None),
            (date(2026, 10, 2), "jp", "A", "brk", -1, 100.0, None),
            (date(2026, 10, 5), "jp", "A", "brk", -1, 100.0, None),
            (date(2026, 10, 6), "jp", "A", "brk", -1, 100.0, None),
        ],
        columns=cols,
    )
    best = pd.DataFrame(
        {
            "market": ["jp"],
            "symbol": ["A"],
            "best_rule": ["brk"],
            "win_rate": [0.7],
            "num_trades": [10],
        }
    )
    base = _loaders(best_history=_best_history([(date(2026, 9, 1), "A", "brk", 0.7)]))
    return ScoringLoaders(
        bars=base.bars,
        rule_signals=lambda market, start: sig,
        prediction_h1=base.prediction_h1,
        prediction_history=base.prediction_history,
        rule_best=lambda market: best,
        rule_best_history=base.rule_best_history,
        rule_outcomes_scored=base.rule_outcomes_scored,
    )


def test_backtest_gap_split_by_side() -> None:
    report, _ = build_rule_accuracy("jp", AS_OF, loaders=_side_loaders())
    gap = report["backtest_gap"]["brk"]
    assert set(gap) == {"buy", "sell"}
    buy, sell = gap["buy"], gap["sell"]
    assert buy["live_hit_rate_h5"] == 1.0
    assert buy["n_h5"] == 1
    assert abs(buy["backtest_win_rate"] - 0.7) < 1e-9
    assert sell["n_h5"] == 3
    assert sell["live_hit_rate_h5"] == 0.0
    # 往復の勝率は買いの取引だけなので、売りの側には並べない
    assert sell["backtest_win_rate"] is None
    for side in (buy, sell):
        assert side["n_unmatched"] == 0
        assert side["approx"] is False
    # 今までの向きをまとめた値は backtest_gap_combined に残る
    comb = report["backtest_gap_combined"]["brk"]
    assert comb["n_h5"] == 4
    assert abs(comb["live_hit_rate_h5"] - 0.25) < 1e-9
    assert abs(comb["backtest_win_rate"] - 0.7) < 1e-9


def test_backtest_gap_side_has_baseline_and_edge() -> None:
    report, _ = build_rule_accuracy("jp", AS_OF, loaders=_side_loaders())
    buy = report["backtest_gap"]["brk"]["buy"]
    sell = report["backtest_gap"]["brk"]["sell"]
    # その日の上がった割合は A・B・C のうち A だけで 1/3、下がった割合は B だけで 1/3
    assert abs(buy["baseline_rate"] - 1 / 3) < 1e-9
    assert abs(buy["edge"] - 2 / 3) < 1e-9
    assert abs(sell["baseline_rate"] - 1 / 3) < 1e-9
    assert abs(sell["edge"] - (-1 / 3)) < 1e-9


def test_rerun_same_output() -> None:
    a, rows_a = build_rule_accuracy("jp", AS_OF, loaders=_loaders())
    b, rows_b = build_rule_accuracy("jp", AS_OF, loaders=_loaders())
    a.pop("generated_at")
    b.pop("generated_at")
    assert a == b
    cols = [c for c in rows_a.columns if c != "scored_at"]
    pd.testing.assert_frame_equal(rows_a[cols], rows_b[cols])


def test_as_of_cuts_future_signals() -> None:
    early = date(2026, 10, 15)
    report, rows = build_rule_accuracy("jp", early, loaders=_loaders())
    assert rows["signal_date"].max() <= early
    assert report["date"] == "2026-10-15"


def test_weekly_diff_rules(tmp_path: Path) -> None:
    hist = _best_history(
        [
            (date(2026, 10, 25), "A", "rsi", 0.6),
            (date(2026, 10, 25), "C", "macd", 0.6),
            (date(2026, 11, 1), "A", "bollinger", 0.7),
            (date(2026, 11, 1), "B", "rsi", 0.6),
        ]
    )
    outcomes = pd.DataFrame(
        {
            "signal_date": [date(2026, 10, 20)] * 2,
            "symbol": ["A", "B"],
            "rule_name": ["rsi", "rsi"],
            "signal": [1, 1],
            "horizon": [5, 5],
            "hit": [True, False],
            "baseline": [0.5, 0.5],
            "signed_return": [0.02, -0.01],
            "scored_at": [datetime(2026, 10, 27, 8, 0)] * 2,
        }
    )
    seen: list[tuple[datetime, datetime]] = []

    def scored(market: str, start: datetime, end: datetime) -> pd.DataFrame:
        seen.append((start, end))
        return outcomes

    loaders = _loaders(best_history=hist)
    loaders = ScoringLoaders(**{**loaders.__dict__, "rule_outcomes_scored": scored})
    # 2026-W44 = JST 10-26（月）0:00 〜 11-02（月）0:00。scored_at は UTC なので 9 時間前にずらす
    path = write_weekly_rule_accuracy("jp", 2026, 44, output_dir=tmp_path, loaders=loaders)
    assert path == tmp_path / "weekly" / "2026-W44.json"
    start = datetime(2026, 10, 25, 15, 0)
    assert seen == [(start, start + timedelta(days=7))]
    data = json.loads(path.read_text(encoding="utf-8"))
    diff = data["rule_best_diff"]
    assert diff["evaluated_on"] == "2026-11-01"
    assert diff["previous_evaluated_on"] == "2026-10-25"
    assert [c["symbol"] for c in diff["changed"]] == ["A"]
    assert diff["changed"][0] == {"symbol": "A", "from": "rsi", "to": "bollinger"}
    assert diff["added"] == [{"symbol": "B", "rule": "rsi"}]
    assert diff["removed"] == [{"symbol": "C", "rule": "macd"}]
    s = data["outcomes"]["rsi"]["buy"]["5"]
    assert s["n"] == 2 and s["hit_rate"] == 0.5
    assert data["n_outcomes"] == 2
