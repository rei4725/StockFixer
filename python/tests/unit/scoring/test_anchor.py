"""錨（anchor）の純粋関数のテスト。"""

import numpy as np
import pandas as pd

from src.scoring.anchor import (
    daily_baseline,
    forward_returns,
    price_frame,
    score_signals,
    summarize,
)


def _fwd(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(symbol, date, ret_5) から score_signals 用の fwd を作る。"""
    df = pd.DataFrame(rows, columns=["symbol", "date", "ret_5"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for h in (1, 20):
        df[f"ret_{h}"] = np.nan
    for h in (1, 5, 20):
        df[f"split_{h}"] = False
    df["px"] = 100.0
    return df


def _sig(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "symbol", "key", "direction"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def test_price_frame_prefers_adj_close() -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "ts": pd.to_datetime(["2026-01-05 09:00", "2026-01-06 09:00"]),
            "close": [100.0, 110.0],
            "adj_close": [50.0, np.nan],
        }
    )
    out = price_frame(raw)
    assert list(out.columns) == ["symbol", "date", "px"]
    assert out["px"].tolist() == [50.0, 110.0]
    assert out["date"].iloc[0].isoformat() == "2026-01-05"


def test_forward_returns_by_row_order() -> None:
    dates = pd.date_range("2026-01-05", periods=5).date
    px = pd.DataFrame({"symbol": "A", "date": dates, "px": [100, 110, 121, 133.1, 146.41]})
    fwd = forward_returns(px, horizons=(3,))
    assert abs(fwd["ret_3"].iloc[0] - 0.331) < 1e-9
    assert abs(fwd["ret_3"].iloc[1] - (146.41 / 110 - 1)) < 1e-9
    assert fwd["ret_3"].iloc[2:].isna().all()


def test_zero_move_is_miss_both_sides() -> None:
    d = "2026-01-05"
    fwd = _fwd([("A", d, 0.1), ("B", d, 0.0), ("C", d, -0.1), ("D", d, 0.2)])
    base = daily_baseline(fwd, 5)
    assert base["up_rate"].iloc[0] == 0.5
    assert base["down_rate"].iloc[0] == 0.25
    scored, _ = score_signals(_sig([(d, "B", "k", 1)]), fwd, horizons=(5,))
    assert bool(scored["hit"].iloc[0]) is False


def test_baseline_is_per_signal_day() -> None:
    d1, d2 = "2026-01-05", "2026-01-06"
    fwd = _fwd([("A", d1, 0.1), ("B", d1, 0.2), ("A", d2, -0.1), ("B", d2, -0.2)])
    sig = _sig([(d1, "A", "k", 1), (d2, "A", "k", 1)])
    scored, _ = score_signals(sig, fwd, horizons=(5,))
    scored = scored.sort_values("date")
    assert scored["baseline"].tolist() == [1.0, 0.0]


def test_sell_hits_on_fall() -> None:
    d = "2026-01-05"
    fwd = _fwd([("A", d, -0.1), ("B", d, 0.1)])
    scored, _ = score_signals(_sig([(d, "A", "k", -1)]), fwd, horizons=(5,))
    row = scored.iloc[0]
    assert bool(row["hit"]) is True
    assert abs(row["signed_return"] - 0.1) < 1e-12
    assert row["baseline"] == 0.5


def test_no_future_bar_not_scored() -> None:
    dates = pd.date_range("2026-01-05", periods=3).date
    px = pd.DataFrame({"symbol": "A", "date": dates, "px": [100.0, 101.0, 102.0]})
    fwd = forward_returns(px)
    sig = _sig([("2026-01-07", "A", "k", 1)])
    scored, _ = score_signals(sig, fwd, horizons=(1,))
    assert scored.empty
    sig2 = _sig([("2026-01-05", "A", "k", 1), ("2026-01-06", "A", "k", 0)])
    scored2, _ = score_signals(sig2, fwd, horizons=(1,))
    assert len(scored2) == 1


def test_split_jump_excluded() -> None:
    dates = pd.date_range("2026-01-05", periods=4).date
    px = pd.DataFrame(
        {
            "symbol": ["A"] * 4 + ["B"] * 4,
            "date": list(dates) * 2,
            "px": [100.0, 100.0, 200.0, 200.0, 100.0, 101.0, 102.0, 103.0],
        }
    )
    fwd = forward_returns(px, horizons=(2,))
    d0 = dates[0].isoformat()
    sig = _sig([(d0, "A", "k", 1), (d0, "B", "k", 1)])
    scored, n_excl = score_signals(sig, fwd, horizons=(2,))
    assert n_excl == 1
    assert scored["symbol"].tolist() == ["B"]


def test_summarize_edge() -> None:
    scored = pd.DataFrame(
        {
            "key": ["k"] * 4,
            "horizon": [5] * 4,
            "hit": [True, True, True, False],
            "baseline": [0.5, 0.5, 0.5, 0.5],
            "signed_return": [0.1, 0.1, 0.1, -0.1],
        }
    )
    out = summarize(scored, ["key", "horizon"])
    row = out.iloc[0]
    assert row["n"] == 4
    assert row["hit_rate"] == 0.75
    assert row["baseline_rate"] == 0.5
    assert row["edge"] == 0.25
    assert abs(row["mean_signed_return"] - 0.05) < 1e-12
