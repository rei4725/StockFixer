"""発注の判定の5日の平均から、引けていない今日の足を落とす（#2112）。"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.trading.execution import params

_CLOSED_DATES = pd.to_datetime(
    ["2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
)
_CLOSED_VOLUME = [900_000.0, 1_000_000.0, 1_100_000.0, 1_200_000.0, 1_300_000.0, 1_400_000.0]
_CLOSED_HIGH = [105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
_CLOSED_LOW = [95.0, 96.0, 97.0, 98.0, 99.0, 100.0]
_CLOSED_CLOSE = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]


def _hist(with_today: bool, today: str = "2026-11-23") -> pd.DataFrame:
    dates = list(_CLOSED_DATES)
    volume = list(_CLOSED_VOLUME)
    high, low, close = list(_CLOSED_HIGH), list(_CLOSED_LOW), list(_CLOSED_CLOSE)
    if with_today:
        dates.append(pd.Timestamp(today))
        volume.append(50_000.0)
        high.append(105.5)
        low.append(105.0)
        close.append(105.2)
    return pd.DataFrame(
        {"Volume": volume, "High": high, "Low": low, "Close": close},
        index=pd.DatetimeIndex(dates),
    )


def _expected_closed_last5() -> tuple[float, float]:
    vol = sum(_CLOSED_VOLUME[-5:]) / 5
    ratios = [(h - l) / c for h, l, c in zip(_CLOSED_HIGH, _CLOSED_LOW, _CLOSED_CLOSE)][-5:]
    return vol, sum(ratios) / 5


def _market_data(hist: pd.DataFrame) -> MagicMock:
    md = MagicMock()
    md.get_ohlcv.return_value = hist
    return md


def _set_now(monkeypatch: pytest.MonkeyPatch, utc: datetime) -> None:
    monkeypatch.setattr(params, "_utcnow", lambda: utc)


def test_jp_before_close_drops_today_intraday_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2026-11-23 09:25 JST = 00:25 UTC。最後の行は寄りから数分の場中の足
    _set_now(monkeypatch, datetime(2026, 11, 23, 0, 25, tzinfo=timezone.utc))
    avg_volume, spread = params._load_execution_metrics("jp", "7203", _market_data(_hist(True)))
    exp_vol, exp_spread = _expected_closed_last5()
    assert avg_volume == pytest.approx(exp_vol)
    assert spread == pytest.approx(exp_spread)


def test_jp_at_850_without_today_row_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # 08:50 JST = 前日 23:50 UTC。今日の行は無い
    _set_now(monkeypatch, datetime(2026, 11, 22, 23, 50, tzinfo=timezone.utc))
    avg_volume, spread = params._load_execution_metrics("jp", "7203", _market_data(_hist(False)))
    exp_vol, exp_spread = _expected_closed_last5()
    assert avg_volume == pytest.approx(exp_vol)
    assert spread == pytest.approx(exp_spread)


def test_jp_after_close_keeps_today_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    # 16:00 JST = 07:00 UTC。今日の足は引けているので残す
    _set_now(monkeypatch, datetime(2026, 11, 23, 7, 0, tzinfo=timezone.utc))
    hist = _hist(True)
    avg_volume, spread = params._load_execution_metrics("jp", "7203", _market_data(hist))
    recent = hist.tail(5)
    assert avg_volume == pytest.approx(float(recent["Volume"].mean()))
    exp_spread = float(((recent["High"] - recent["Low"]) / recent["Close"]).mean())
    assert spread == pytest.approx(exp_spread)


def test_us_judged_by_us_close(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2026-11-23 16:00 JST = 07:00 UTC は jp の引けの後だが、ニューヨークは 11-23 02:00 で
    # 11-23 の us の足はまだ引けていない
    _set_now(monkeypatch, datetime(2026, 11, 23, 7, 0, tzinfo=timezone.utc))
    avg_volume, _ = params._load_execution_metrics("us", "AAPL", _market_data(_hist(True)))
    exp_vol, _ = _expected_closed_last5()
    assert avg_volume == pytest.approx(exp_vol)


def test_us_after_us_close_keeps_today_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2026-11-23 17:00 EST = 22:00 UTC。us の引けの後なので今日の足を残す
    _set_now(monkeypatch, datetime(2026, 11, 23, 22, 0, tzinfo=timezone.utc))
    hist = _hist(True)
    avg_volume, _ = params._load_execution_metrics("us", "AAPL", _market_data(hist))
    assert avg_volume == pytest.approx(float(hist.tail(5)["Volume"].mean()))


def test_tz_aware_index_drops_today(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_now(monkeypatch, datetime(2026, 11, 23, 0, 25, tzinfo=timezone.utc))
    hist = _hist(True)
    hist.index = hist.index.tz_localize("Asia/Tokyo")
    avg_volume, _ = params._load_execution_metrics("jp", "7203", _market_data(hist))
    exp_vol, _ = _expected_closed_last5()
    assert avg_volume == pytest.approx(exp_vol)
