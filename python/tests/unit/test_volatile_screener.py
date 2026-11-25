"""backtest/screener.py の銘柄一覧の読みと、値動きの大きい銘柄の選び方のテスト (#2175)"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.backtest import screener

_PYTHON_ROOT = Path(__file__).resolve().parents[2]


def _bars(days: int, close: float, width: float, volume: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "High": [close + width / 2] * days,
            "Low": [close - width / 2] * days,
            "Close": [close] * days,
            "Volume": [volume] * days,
        },
        index=pd.date_range("2024-01-01", periods=days),
    )


class _FakeDataPort:
    """銘柄ごとに決めた足を返すデータの口。"""

    def __init__(self, frames: dict) -> None:
        self._frames = frames
        self.calls: list[tuple] = []

    def download(self, ticker, *, start=None, end=None):
        self.calls.append((ticker, start, end))
        value = self._frames[ticker]
        if isinstance(value, Exception):
            raise value
        return value

    def add_technical_indicators(self, df):
        return df.assign(atr=df["High"] - df["Low"])


@pytest.fixture
def use_watchlist(tmp_path, monkeypatch):
    """銘柄の一覧を tmp のファイルに差し替える。"""

    def _use(data: dict) -> None:
        path = tmp_path / "watchlist.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(screener, "get_watchlist_path", lambda: str(path))

    return _use


def test_load_watchlist_symbols_reads_shared_config():
    """パスを差し替えなくても python/config/watchlist.json の並びを読む。"""
    path = _PYTHON_ROOT / "config" / "watchlist.json"
    expected = json.loads(path.read_text(encoding="utf-8"))["jp"]

    assert expected
    assert screener._load_watchlist_symbols("jp") == expected


def test_load_watchlist_symbols_unknown_market_is_empty(use_watchlist):
    use_watchlist({"jp": ["7203", "6758"]})

    assert screener._load_watchlist_symbols("jp") == ["7203", "6758"]
    assert screener._load_watchlist_symbols("us") == []


@pytest.mark.parametrize(
    ("top_n", "expected"),
    [(3, ["BOTH", "WIDE", "MID"]), (10, ["BOTH", "WIDE", "MID", "BUSY"])],
)
def test_screen_volatile_symbols_orders_by_score(use_watchlist, monkeypatch, top_n, expected):
    """スコア（出来高の順位 0.4・ATR% の順位 0.6）の降順で top_n 件を返す。"""
    # スコア: BOTH 0.85 / WIDE 0.70 / MID 0.50 / BUSY 0.45
    frames = {
        "BUSY": _bars(100, close=2000.0, width=20.0, volume=3000.0),
        "SHORT": _bars(99, close=100.0, width=50.0, volume=9000.0),
        "WIDE": _bars(100, close=100.0, width=8.0, volume=1000.0),
        "ERROR": RuntimeError("取得に失敗"),
        "MID": _bars(100, close=100.0, width=3.0, volume=2000.0),
        "NONE": None,
        "BOTH": _bars(100, close=1000.0, width=50.0, volume=4000.0),
    }
    use_watchlist({"us": list(frames)})
    port = _FakeDataPort(frames)
    monkeypatch.setattr(screener, "get_backtest_data_port", lambda: port)

    result = screener.screen_volatile_symbols(
        "us", top_n=top_n, screen_start="2024-01-01", screen_end="2024-06-30"
    )

    assert result == expected
    assert port.calls == [(symbol, "2024-01-01", "2024-06-30") for symbol in frames]
