"""ユニットテスト: レジームレバレッジ戦略の週次の回は、途中の足で判定せず投げる(#2149)。

最後の足(2026-11-06 金曜)の手前は終値500ドルで、200日線はほぼ500ドルになる。最後の足の
終値を510ドルにすると線の上、490ドルにすると線の下になる。
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check
from src.trading.regime_leverage_strategy.types import RegimeLeverageSnapshot

_SERVICE = "src.trading.regime_leverage_strategy.service"
_JST = ZoneInfo("Asia/Tokyo")
_FRIDAY = "2026-11-06"
_SCHEDULED = datetime(2026, 11, 7, 6, 30, tzinfo=_JST)
_RECOVERY = datetime(2026, 11, 7, 10, 30, tzinfo=_JST)
_ABOVE = 510.0
_BELOW = 490.0
_NORMAL_VOLUME = 1_000_000.0
_PARTIAL_VOLUME = _NORMAL_VOLUME * 0.15


def _frame(close: float, volume: float) -> pd.DataFrame:
    """金曜までの日足。close と volume は金曜の足の値"""
    idx = pd.bdate_range(end=_FRIDAY, periods=260)
    df = pd.DataFrame(
        {"Open": 500.0, "High": 515.0, "Low": 485.0, "Close": 500.0, "Volume": _NORMAL_VOLUME},
        index=idx,
    )
    df.loc[idx[-1], ["Close", "Volume"]] = [close, volume]
    return df


def _holding() -> RegimeLeverageSnapshot:
    return RegimeLeverageSnapshot(
        id=1,
        executed_at=datetime(2026, 11, 5),
        action="noop",
        reason="daily_noop",
        spy_price_usd=500.0,
        usdjpy_rate=146.0,
        shares=27.0,
        entry_date=datetime(2026, 6, 1),
        entry_price_jpy=72500.0,
        entry_commission_jpy=0.0,
        equity_at_entry_jpy=1_000_000.0,
        stop_price_jpy=65000.0,
        equity_now_jpy=1_000_000.0,
        maintenance_ratio=None,
    )


class _Book:
    """週次の回を続けて動かし、書いた行を覚えておく"""

    def __init__(self, held: bool) -> None:
        self.snapshot = _holding() if held else None
        self.rows: list = []

    def run(self, df: pd.DataFrame, moment: datetime = _SCHEDULED):
        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[no-untyped-def, override]
                if tz is not None:
                    return moment.astimezone(tz)
                return moment.astimezone(timezone.utc).replace(tzinfo=None)

        port = MagicMock()
        port.get_stock_data.return_value = df
        port.get_forex_data.return_value = pd.DataFrame(
            {"Close": [146.0]}, index=[pd.Timestamp(_FRIDAY)]
        )
        with (
            patch(f"{_SERVICE}.datetime", _Frozen),
            patch(f"{_SERVICE}.get_latest_snapshot", return_value=self.snapshot),
            patch(f"{_SERVICE}.insert_snapshot", side_effect=self.rows.append),
        ):
            return run_regime_leverage_weekly_check(port)


# 保有の有無・終値と、ふつうの足のときの判定
_CASES = [
    (True, _ABOVE, "weekly_noop"),
    (True, _BELOW, "regime_flip"),
    (False, _ABOVE, "regime_entry"),
    (False, _BELOW, "weekly_noop"),
]


@pytest.mark.parametrize("held, close, _reason", _CASES)
def test_partial_bar_raises_without_row(held, close, _reason):
    """途中の足なら、転換・建玉・何もしないのどれでも行を書かず、日付と比を添えて投げること"""
    book = _Book(held)

    with pytest.raises(RuntimeError, match=f"last_bar={_FRIDAY}, volume_ratio=0.150"):
        book.run(_frame(close, _PARTIAL_VOLUME))

    assert book.rows == []


def test_zero_volume_bar_raises_without_row():
    """出来高0の足も途中の足として投げること"""
    book = _Book(held=True)

    with pytest.raises(RuntimeError, match="volume_ratio=0.000"):
        book.run(_frame(_BELOW, 0.0))

    assert book.rows == []


@pytest.mark.parametrize("held, close, reason", _CASES)
def test_next_run_judges_corrected_close(held, close, reason):
    """次の呼び出しで同じ日の足が直っていれば、途中の終値でなく直った終値で判定すること"""
    book = _Book(held)
    partial_close = _BELOW if close == _ABOVE else _ABOVE

    with pytest.raises(RuntimeError, match="volume_ratio="):
        book.run(_frame(partial_close, _PARTIAL_VOLUME))
    decision = book.run(_frame(close, _NORMAL_VOLUME), _RECOVERY)

    assert decision.reason == reason
    assert decision.spy_price_usd == close
    assert book.rows == [decision]
    if reason == "regime_entry":
        assert decision.entry_price_jpy == pytest.approx(close * 146.0, rel=0.01)


@pytest.mark.parametrize("ratio", [0.4, 0.6, 1.0, 2.0, np.nan])
@pytest.mark.parametrize("held, close, reason", _CASES)
def test_normal_half_day_and_missing_volume_write_row(held, close, reason, ratio):
    """ふつう・半日立会いの程度・欠けの出来高なら、今までどおり判定して行を書くこと"""
    book = _Book(held)

    decision = book.run(_frame(close, _NORMAL_VOLUME * ratio))

    assert decision.reason == reason
    assert book.rows == [decision]


@pytest.mark.parametrize("held, close, reason", _CASES)
def test_missing_previous_volumes_write_row(held, close, reason):
    """比べる手前の出来高が欠けていれば、今までどおり判定して行を書くこと"""
    book = _Book(held)
    df = _frame(close, _PARTIAL_VOLUME)
    df.iloc[:-1, df.columns.get_loc("Volume")] = np.nan

    decision = book.run(df)

    assert decision.reason == reason
    assert book.rows == [decision]


@pytest.mark.parametrize("held, close, reason", _CASES)
def test_no_volume_column_writes_row(held, close, reason):
    """出来高の列が無ければ、今までどおり判定して行を書くこと"""
    book = _Book(held)

    decision = book.run(_frame(close, _PARTIAL_VOLUME).drop(columns="Volume"))

    assert decision.reason == reason
    assert book.rows == [decision]
