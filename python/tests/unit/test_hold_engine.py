"""hold_engine（保有/撤退エンジン）の単体テスト。

合成価格系列を pd.DataFrame で作り、各ルールを個別に検証する。
ルックアヘッドが無いこと（その日までの情報のみで判定）も確認する。
"""

import numpy as np
import pandas as pd
import pytest

from src.screening.hold_engine import simulate_position
from src.screening.types import HoldRules, PositionEvent


def _make_df(closes) -> pd.DataFrame:
    """Close 系列から date 列付き DataFrame を作る（昇順）。"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
            "Close": closes,
        }
    )


def _entry_date(df: pd.DataFrame) -> str:
    return str(df["date"].iloc[0])


def test_monotonic_uptrend_holds_to_end():
    """単調上昇（利確オフ）→ 末尾まで保有し最後に exit、multiple が正しい。"""
    df = _make_df(np.linspace(10.0, 30.0, 100))
    rules = HoldRules(scale_out_multiples=[])  # 放置
    events = simulate_position(df, _entry_date(df), rules)

    assert events[0].action == "entry"
    assert events[0].held_fraction == 1.0
    assert events[-1].action == "exit"
    assert events[-1].reason == "end_of_data"
    assert events[-1].held_fraction == 0.0
    # entry=10, end=30 → multiple=3.0
    assert events[-1].multiple == 3.0
    # 利確オフなので途中イベントは無い
    assert [e.action for e in events] == ["entry", "exit"]


def test_ma_break_exits_on_break_day():
    """40週線割れ → その日に exit（reason="ma_break"）。"""
    # 200日線が成立するよう十分な長さの平坦系列を作り、最後に MA を割る。
    closes = np.full(206, 100.0)
    closes[-1] = 99.0
    df = _make_df(closes)
    events = simulate_position(df, _entry_date(df))

    assert events[-1].action == "exit"
    assert events[-1].reason == "ma_break"
    assert events[-1].date == str(df["date"].iloc[-1])
    # 撤退後に後続イベントは無い
    assert events[-1] is events[-1]
    assert all(e.reason != "trail_stop" for e in events)


def test_trail_stop_exits_on_drawdown():
    """高値から -35% 下落 → exit（reason="trail_stop"）。"""
    # 100→150 に上げてから 90 へ急落（150*0.65=97.5 を割る）。MA は窓未満で不発。
    closes = np.concatenate([np.linspace(100.0, 150.0, 50), np.linspace(149.0, 90.0, 30)])
    df = _make_df(closes)
    events = simulate_position(df, _entry_date(df))

    exit_events = [e for e in events if e.action == "exit"]
    assert len(exit_events) == 1
    assert exit_events[0].reason == "trail_stop"
    assert exit_events[0].held_fraction == 0.0
    # 利確倍率（2x/5x）には届いていない
    assert all(e.action != "scale_out" for e in events)


def test_scale_out_at_2x_and_5x():
    """2倍/5倍到達 → scale_out が1回ずつ、held_fraction が 1.0→0.8→0.6。"""
    df = _make_df(np.linspace(10.0, 60.0, 100))  # entry=10 → 最大6倍
    events = simulate_position(df, _entry_date(df))

    scale_events = [e for e in events if e.action == "scale_out"]
    assert len(scale_events) == 2
    assert scale_events[0].reason == "scale_2x"
    assert scale_events[0].held_fraction == pytest.approx(0.8)
    assert scale_events[1].reason == "scale_5x"
    assert scale_events[1].held_fraction == pytest.approx(0.6)
    # 倍率の確認（到達時の multiple は対象倍率以上）
    assert scale_events[0].multiple >= 2.0
    assert scale_events[1].multiple >= 5.0
    # 最終的に手仕舞い
    assert events[-1].action == "exit"
    assert events[-1].held_fraction == 0.0


def test_no_scale_out_when_multiples_empty():
    """scale_out_multiples=[] → 利確イベントが出ない（放置）。"""
    df = _make_df(np.linspace(10.0, 60.0, 100))
    rules = HoldRules(scale_out_multiples=[])
    events = simulate_position(df, _entry_date(df), rules)

    assert all(e.action != "scale_out" for e in events)
    assert [e.action for e in events] == ["entry", "exit"]


def test_no_scale_out_when_fraction_zero():
    """scale_out_fraction=0 → 利確しない（放置）。"""
    df = _make_df(np.linspace(10.0, 60.0, 100))
    rules = HoldRules(scale_out_fraction=0.0)
    events = simulate_position(df, _entry_date(df), rules)

    assert all(e.action != "scale_out" for e in events)


def test_early_crash_exits_immediately():
    """エントリー直後に急落 → 早期 exit、後続イベントが無い。"""
    closes = np.array([100.0, 60.0, 70.0, 80.0])  # 1日目で -40%
    df = _make_df(closes)
    events = simulate_position(df, _entry_date(df))

    assert len(events) == 2
    assert events[0].action == "entry"
    assert events[1].action == "exit"
    assert events[1].reason == "trail_stop"
    assert events[1].date == str(df["date"].iloc[1])


def test_no_lookahead():
    """ルックアヘッドが無いこと: 撤退日で系列を打ち切っても結果が同一。"""
    closes = np.concatenate(
        [np.linspace(100.0, 150.0, 50), np.linspace(149.0, 90.0, 30), np.full(20, 200.0)]
    )
    df_full = _make_df(closes)
    entry = _entry_date(df_full)
    full_events = simulate_position(df_full, entry)

    exit_idx = next(i for i, e in enumerate(full_events) if e.action == "exit")
    exit_date = full_events[exit_idx].date

    # 撤退日までで系列を切っても、その日までの判定は変わらないはず
    cutoff = df_full.index[df_full["date"] == exit_date][0]
    df_trunc = df_full.iloc[: cutoff + 1].copy()
    trunc_events = simulate_position(df_trunc, entry)

    assert [(e.action, e.reason, e.date) for e in trunc_events] == [
        (e.action, e.reason, e.date) for e in full_events[: exit_idx + 1]
    ]


def test_missing_entry_date_returns_empty():
    """entry_date が系列に無ければ空リストを返す。"""
    df = _make_df(np.linspace(10.0, 20.0, 30))
    assert simulate_position(df, "1999-01-01") == []


def test_event_types_are_position_event():
    """返り値は PositionEvent のリスト。"""
    df = _make_df(np.linspace(10.0, 30.0, 50))
    events = simulate_position(df, _entry_date(df))
    assert events
    assert all(isinstance(e, PositionEvent) for e in events)
