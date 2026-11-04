"""値動きの量と損切りの線の比べ（#1547）。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtest import volatility_compare
from src.backtest.pipeline.runner import PreparedBacktest
from src.backtest.volatility_compare import (
    run_volatility_sizing_compare,
    write_compare_result,
)
from src.domain.types import SymbolTask

_START = "2020-01-01"
_END = "2025-01-01"


def _prepared(bars: int = 200, warmup: int = 60, seed: int = 1) -> PreparedBacktest:
    """warmup 本の助走のあと、20日ごとに建てては決める短い作りの値。"""
    rng = np.random.default_rng(seed)
    closes = 1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, bars)))
    index = pd.date_range("2024-01-01", periods=bars, freq="B")
    df = pd.DataFrame({"Open": closes, "Close": closes, "Volume": 1_000_000.0}, index=index)
    sim_df = df.iloc[warmup:]
    signal = pd.Series(0, index=sim_df.index)
    signal.iloc[::20] = 1
    signal.iloc[10::20] = -1
    return PreparedBacktest(
        sim_df=sim_df,
        test_df=sim_df,
        closes=df["Close"],
        signal=signal,
        pred=signal * 0.01,
        model_manager=None,
    )


def _run(tasks: list[SymbolTask], prepare: Callable[..., PreparedBacktest]) -> dict[str, dict]:
    with patch.object(volatility_compare, "prepare_backtest_signals", side_effect=prepare):
        return run_volatility_sizing_compare(tasks, _START, _END)["markets"]


def _form(result: dict, name: str, k: float | None = None) -> dict:
    return next(f for f in result["forms"] if f["form"] == name and f["k"] == k)


def test_returns_all_forms_and_ks() -> None:
    """4つの形と3つの k の結果があり、形ごとの数字がそろう"""
    markets = _run([SymbolTask(market="us", symbol="AAA")], lambda *a, **k: _prepared())

    forms = markets["us"]["forms"]
    assert [(f["form"], f["k"]) for f in forms] == [
        ("fixed", None),
        ("size", None),
        ("stop", 2.0),
        ("stop", 3.0),
        ("stop", 4.0),
        ("both", 2.0),
        ("both", 3.0),
        ("both", 4.0),
    ]
    for form in forms:
        assert form["symbols"] == 1
        assert form["num_trades"] > 0
        assert form["max_drawdown"] is not None
        assert form["return_over_volatility"] is not None
    assert markets["us"]["period"]["start"] == "2024-03-25"
    assert markets["us"]["symbols_lacking_forecast"] == 0


def test_stop_multiple_changes_how_often_the_stop_cuts() -> None:
    """k が小さいほど線が近く、損切りで刈られる回数が増える"""
    markets = _run([SymbolTask(market="us", symbol="AAA")], lambda *a, **k: _prepared())
    result = markets["us"]

    assert _form(result, "stop", 2.0)["stop_outs"] > _form(result, "stop", 4.0)["stop_outs"]
    # size は線を変えない: 刈られるのは fixed の線と同じ足
    assert _form(result, "size")["stop_outs"] == _form(result, "fixed")["stop_outs"]


def test_symbol_without_enough_values_falls_back_and_is_counted() -> None:
    """値の足りない銘柄は固定の扱いに倒れ、数が結果に残る"""

    def short_history(*_args: object, **_kwargs: object) -> PreparedBacktest:
        prepared = _prepared(bars=19, warmup=0)
        signal = pd.Series(0, index=prepared.sim_df.index)
        signal.iloc[0] = 1
        return PreparedBacktest(
            prepared.sim_df, prepared.test_df, prepared.closes, signal, signal * 0.01, None
        )

    markets = _run([SymbolTask(market="jp", symbol="1111")], short_history)
    result = markets["jp"]

    assert result["symbols_lacking_forecast"] == 1
    fixed = _form(result, "fixed")
    for name, k in (("size", None), ("stop", 3.0), ("both", 3.0)):
        form = _form(result, name, k)
        assert form["total_return"] == fixed["total_return"]
        assert form["num_trades"] == fixed["num_trades"]


def test_failed_symbol_is_counted_and_the_rest_continue() -> None:
    def prepare(market: str, symbol: str, **_kwargs: object) -> PreparedBacktest:
        if symbol == "BAD":
            raise ValueError("no data")
        return _prepared()

    markets = _run(
        [SymbolTask(market="us", symbol="BAD"), SymbolTask(market="us", symbol="OK")], prepare
    )

    assert markets["us"]["symbols_failed"] == 1
    assert markets["us"]["symbols"] == 1
    assert _form(markets["us"], "fixed")["symbols"] == 1


def test_symbols_are_capped_per_market() -> None:
    tasks = [SymbolTask(market="us", symbol=f"S{i}") for i in range(5)]
    seen: list[str] = []

    def prepare(market: str, symbol: str, **_kwargs: object) -> PreparedBacktest:
        seen.append(symbol)
        return _prepared(bars=80, warmup=30)

    with patch.object(volatility_compare, "MAX_SYMBOLS_PER_MARKET", 2):
        markets = _run(tasks, prepare)

    assert seen == ["S0", "S1"]
    assert markets["us"]["symbols_target"] == 2
    assert markets["us"]["symbols_over_limit"] == 3


def test_no_symbol_compared_raises() -> None:
    def prepare(*_args: object, **_kwargs: object) -> PreparedBacktest:
        raise ValueError("no data")

    with pytest.raises(RuntimeError, match="0件"):
        _run([SymbolTask(market="us", symbol="BAD")], prepare)


def test_result_is_written_as_strict_json(tmp_path: Path) -> None:
    with patch.object(
        volatility_compare, "prepare_backtest_signals", side_effect=lambda *a, **k: _prepared()
    ):
        result = run_volatility_sizing_compare(
            [SymbolTask(market="us", symbol="AAA")], _START, _END
        )

    path = write_compare_result(result, date(2026, 11, 8), str(tmp_path / "out"))

    assert Path(path).name == "2026-W45.json"
    text = Path(path).read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text)["week"] == "2026-W45"
