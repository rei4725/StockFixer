"""backtest_promotion.holdout_rows（holdout と old_production が使う行の部品）のテスト。"""

import math

import pandas as pd
import pytest

from src.prediction.backtest_promotion.holdout_rows import (
    both_metrics,
    close_by_day,
    daily_metrics,
    with_market_day,
)


def test_close_by_day_drops_tz_and_time():
    index = pd.DatetimeIndex(["2026-11-24 15:00", "2026-11-25 15:00"], tz="Asia/Tokyo", name="Date")
    ohlcv = pd.DataFrame({"Close": [100.0, 101.5]}, index=index)

    assert close_by_day(ohlcv) == {
        pd.Timestamp("2026-11-24"): 100.0,
        pd.Timestamp("2026-11-25"): 101.5,
    }


def test_close_by_day_skips_nan_close():
    index = pd.to_datetime(["2026-11-24", "2026-11-25", "2026-11-26"])
    ohlcv = pd.DataFrame({"Close": [100.0, math.nan, 102.0]}, index=index)

    result = close_by_day(ohlcv)

    assert result == {pd.Timestamp("2026-11-24"): 100.0, pd.Timestamp("2026-11-26"): 102.0}


@pytest.mark.parametrize(
    "ohlcv",
    [
        None,
        pd.DataFrame(),
        pd.DataFrame({"Open": [1.0]}, index=pd.to_datetime(["2026-11-24"])),
    ],
)
def test_close_by_day_empty_inputs(ohlcv):
    assert close_by_day(ohlcv) == {}


def test_with_market_day_keys_market_and_day():
    rows = pd.DataFrame({"market": ["jp", "us"], "date": ["2026-11-24", "2026-11-24"]})

    result = with_market_day(rows)

    assert list(result["date"]) == ["jp:2026-11-24", "us:2026-11-24"]
    # 元の表は書き換えない
    assert list(rows["date"]) == ["2026-11-24", "2026-11-24"]


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": ["jp", "jp"],
            "date": ["2026-11-24", "2026-11-24"],
            "actual_ratio": [0.01, -0.01],
            "candidate_ratio": [0.02, -0.02],
            "production_ratio": [-0.02, 0.02],
        }
    )


def test_both_metrics_uses_each_side_ratio():
    candidate, production = both_metrics(_rows())

    assert candidate["n_baseline"] == 2
    assert production["n_baseline"] == 2
    assert candidate["hit_rate_minus_majority"] == pytest.approx(0.5)
    assert production["hit_rate_minus_majority"] == pytest.approx(-0.5)


def test_daily_metrics_per_day_and_missing_day():
    daily = daily_metrics(_rows(), ["2026-11-24", "2026-11-25"])

    assert daily[0]["date"] == "2026-11-24"
    assert daily[0]["candidate"] == pytest.approx(0.5)
    assert daily[0]["production"] == pytest.approx(-0.5)
    assert daily[0]["diff"] == pytest.approx(1.0)
    assert daily[1] == {
        "date": "2026-11-25",
        "candidate": None,
        "production": None,
        "diff": None,
    }
