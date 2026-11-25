"""ルールのシグナルを出せなかった理由の分け方と、止める文の理由ごとの件数の検証（#2173）。"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.orchestration.jobs.unavailable_gates import _raise_if_rule_signals_mostly_unavailable
from src.rule_engine.pipeline import generate_rule_signals
from src.rule_engine.types import RuleSignalBatch, UnavailableRuleSymbol

_SIGNAL_DATE = date(2024, 3, 22)


def _bars(end: str = "2024-03-22", n: int = 60) -> pd.DataFrame:
    close = np.linspace(1000, 1100, n)
    return pd.DataFrame(
        {
            "Open": close - 5,
            "High": close + 10,
            "Low": close - 10,
            "Close": close,
            "Volume": np.full(n, 500_000, dtype=float),
        },
        index=pd.bdate_range(end=end, periods=n),
    )


def _bad_close(value: float) -> pd.DataFrame:
    df = _bars()
    df.iloc[-1, df.columns.get_loc("Close")] = value
    return df


class _Port:
    """銘柄ごとに決めた足を返す（例外を置いた銘柄では投げる）。"""

    def __init__(self, bars: dict[str, pd.DataFrame | Exception | None]):
        self._bars = bars

    def get_ohlcv_with_indicators(self, ticker, lookback_days):
        value = self._bars[ticker.split(".")[0]]
        if isinstance(value, Exception):
            raise value
        return value


def _generate(rules: dict[str, str], port) -> tuple[RuleSignalBatch, MagicMock]:
    effective = pd.DataFrame(
        [
            {"symbol": s, "best_rule": rule, "win_rate": 0.6, "net_profit": 1}
            for s, rule in rules.items()
        ]
    )
    with (
        patch("src.rule_engine.pipeline.load_effective_rules", return_value=effective),
        patch("src.rule_engine.pipeline.upsert_rule_signal") as upsert,
    ):
        batch = generate_rule_signals("jp", signal_date=_SIGNAL_DATE, market_data_port=port)
    return batch, upsert


@pytest.mark.parametrize(
    ("rule", "bars", "reason"),
    [
        ("unknown_rule", _bars(), "未知のルール"),
        ("rsi_contrarian", None, "データ不足"),
        ("rsi_contrarian", pd.DataFrame(), "データ不足"),
        ("rsi_contrarian", _bars(n=19), "データ不足"),
        ("rsi_contrarian", _bars(end="2024-03-01"), "足の停止"),
        ("rsi_contrarian", _bad_close(np.nan), "終値の欠け"),
        ("rsi_contrarian", _bad_close(0.0), "終値の欠け"),
    ],
)
def test_reason_is_split(rule, bars, reason):
    batch, upsert = _generate({"7203": rule}, _Port({"7203": bars}))
    assert batch.unavailable == [UnavailableRuleSymbol("7203", rule, reason)]
    assert batch.signals == []
    upsert.assert_not_called()


def test_reasons_do_not_leak_between_symbols():
    """5つの理由と出せた銘柄が混ざっても、銘柄ごとの理由と並びが保たれる。"""
    rules = {
        "1001": "unknown_rule",
        "1002": "rsi_contrarian",
        "1003": "rsi_contrarian",
        "1004": "rsi_contrarian",
        "1005": "rsi_contrarian",
        "1006": "rsi_contrarian",
    }
    port = _Port(
        {
            "1002": None,
            "1003": _bars(end="2024-03-01"),
            "1004": RuntimeError("fetch error"),
            "1005": _bars(),
            "1006": _bad_close(np.nan),
        }
    )
    batch, upsert = _generate(rules, port)
    assert [(u.symbol, u.reason) for u in batch.unavailable] == [
        ("1001", "未知のルール"),
        ("1002", "データ不足"),
        ("1003", "足の停止"),
        ("1004", "例外"),
        ("1006", "終値の欠け"),
    ]
    assert [s["symbol"] for s in batch.signals] == ["1005"]
    assert batch.total == 6
    upsert.assert_called_once()


def test_reason_falls_back_when_not_given():
    """理由が渡らなければ今までどおり「シグナルなし」にする。"""
    with patch("src.rule_engine.pipeline._get_today_signal", return_value=None):
        batch, _ = _generate({"7203": "rsi_contrarian"}, MagicMock())
    assert [u.reason for u in batch.unavailable] == ["シグナルなし"]


def _batch(reasons: list[str], total: int) -> RuleSignalBatch:
    return RuleSignalBatch(
        unavailable=[
            UnavailableRuleSymbol(f"S{i}", "rsi_contrarian", reason)
            for i, reason in enumerate(reasons)
        ],
        total=total,
    )


def test_error_text_has_counts_by_reason():
    """件数の多い順、同数は理由の名前の順に並ぶ。"""
    reasons = ["足の停止", "例外", "データ不足", "終値の欠け", "例外", "足の停止", "例外"]
    with (
        patch("config.settings.DAILY_FETCH_MAX_UNAVAILABLE_RATIO", 0.5),
        pytest.raises(RuntimeError) as err,
    ):
        _raise_if_rule_signals_mostly_unavailable(_batch(reasons, 10))
    assert type(err.value) is RuntimeError
    assert str(err.value) == (
        "ルールシグナルを出せなかった銘柄が多すぎる: 7 / 対象 10（線 50%）"
        "。理由: 例外 3 件, 足の停止 2 件, データ不足 1 件, 終値の欠け 1 件"
    )


@pytest.mark.parametrize("n", [0, 4, 5])
def test_within_line_does_not_raise(n):
    with patch("config.settings.DAILY_FETCH_MAX_UNAVAILABLE_RATIO", 0.5):
        _raise_if_rule_signals_mostly_unavailable(_batch(["例外"] * n, 10))
