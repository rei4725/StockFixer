"""差分取得で重ねて取った足で、DB に残った途中の足を直すテスト（#2140）"""

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from tests.unit.macro_frames_stub import additional_macro_frame, cross_asset_frame

from src.market_data.pipeline import fetch_stock_data_with_features, run_batch_pipeline

_P = "src.market_data.pipeline"
_COLS = ["Open", "High", "Low", "Close", "Volume"]
_START, _END = "2026-07-01", "2026-10-22"
_LATEST = pd.Timestamp("2026-10-21")  # DB の最新の足
_NEW = pd.Timestamp("2026-10-22")  # 新しい足
_PARTIAL = pd.Timestamp("2026-10-19")  # 途中の足


def _true_bars() -> pd.DataFrame:
    dates = pd.bdate_range(_START, _END)
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 120, n),
            "High": np.linspace(101, 121, n),
            "Low": np.linspace(99, 119, n),
            "Close": np.linspace(100, 120, n),
            "Volume": np.linspace(1_000_000.0, 1_100_000.0, n),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=dates,
    )


def _db_bars(day: pd.Timestamp = _PARTIAL, volume_rate: float = 0.15) -> pd.DataFrame:
    """day の足だけが途中の値（終値 1% ずれ、出来高は本当の volume_rate 倍）の DB の足"""
    bars = _true_bars().loc[:_LATEST, _COLS]
    bars.loc[day, ["Open", "High", "Low", "Close"]] *= 0.99
    bars.loc[day, "Volume"] *= volume_rate
    return bars


def _run(db, yf_bars, defer=True, latest_only_overlap=False):
    """日次の取得を動かし、(結果, 保存の呼び出し, 特徴量の元の表) を返す"""
    if latest_only_overlap:
        yf_bars = yf_bars.loc[:_LATEST]

    def _get(_market, _ticker, start, end):
        return yf_bars.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()

    from src.market_data.technical import add_technical_indicators

    with ExitStack() as stack:
        stack.enter_context(patch(f"{_P}.resolve_article", return_value=None))
        stack.enter_context(patch(f"{_P}.fetch_pageview_features", return_value=None))
        stack.enter_context(patch(f"{_P}.fetch_news_sentiment_with_llm", return_value=None))
        stack.enter_context(
            patch(f"{_P}.fetch_additional_macro_features", side_effect=additional_macro_frame)
        )
        stack.enter_context(
            patch(f"{_P}.fetch_cross_asset_features", side_effect=cross_asset_frame)
        )
        stack.enter_context(patch(f"{_P}.should_fetch_fresh_data", return_value=True))
        stack.enter_context(patch(f"{_P}.drop_unclosed_bars", side_effect=lambda df, _m: df))
        stack.enter_context(patch(f"{_P}.get_raw_ohlcv_from_db", return_value=db))
        mock_get = stack.enter_context(patch(f"{_P}.get_stock_data", side_effect=_get))
        mock_save = stack.enter_context(patch(f"{_P}.save_raw_ohlcv"))
        mock_tech = stack.enter_context(
            patch(f"{_P}.add_technical_indicators", side_effect=add_technical_indicators)
        )
        result = fetch_stock_data_with_features(
            "us", "AAPL", start_date=_START, end_date=_END, defer_raw_save=defer
        )
    base = mock_tech.call_args.args[0] if mock_tech.called else None
    return result, mock_save, base, mock_get


def _saved(result, mock_save, defer):
    """保存の対象の足。どちらの道でも1回だけであることも確かめる"""
    if defer:
        mock_save.assert_not_called()
        return result[3]
    mock_save.assert_called_once()
    assert result[3] is None
    return mock_save.call_args.args[2]


@pytest.mark.parametrize("defer", [True, False])
@pytest.mark.parametrize("day", [_PARTIAL, _LATEST])
def test_partial_bar_is_overwritten_and_saved_once_with_new_bar(defer, day):
    true_bars = _true_bars()
    result, mock_save, base, _ = _run(_db_bars(day), true_bars, defer)

    raw = _saved(result, mock_save, defer)
    assert list(raw.index) == [day, _NEW]
    pd.testing.assert_frame_equal(
        raw[_COLS], true_bars.loc[[day, _NEW], _COLS], check_freq=False, check_names=False
    )
    # 特徴量の元の表にも、取り直した足が1行だけ残る
    assert base.index.is_unique
    assert base.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(
        base[_COLS], true_bars[_COLS], check_freq=False, check_names=False
    )


@pytest.mark.parametrize("volume_rate", [1.0, 0.95, 0.6])
@pytest.mark.parametrize("defer", [True, False])
def test_small_or_no_volume_difference_saves_only_new_bar(defer, volume_rate):
    """終値が配当の調整で 1% ずれていても、出来高が近ければ新しい足だけ"""
    db = _db_bars(volume_rate=volume_rate)
    result, mock_save, base, _ = _run(db, _true_bars(), defer)

    assert list(_saved(result, mock_save, defer).index) == [_NEW]
    assert base.loc[_PARTIAL, "Close"] == db.loc[_PARTIAL, "Close"]


@pytest.mark.parametrize("bad_volume", [np.nan, 0.0, 1000.0])
def test_missing_zero_or_smaller_fresh_volume_does_not_overwrite(bad_volume):
    db, fresh = _db_bars(), _true_bars()
    fresh.loc[_PARTIAL, "Volume"] = bad_volume
    result, _, base, _ = _run(db, fresh)

    assert list(result[3].index) == [_NEW]
    assert base.loc[_PARTIAL, "Volume"] == db.loc[_PARTIAL, "Volume"]


def test_missing_fresh_price_does_not_overwrite():
    fresh = _true_bars()
    fresh.loc[_PARTIAL, "High"] = np.nan
    result, _, _, _ = _run(_db_bars(), fresh)

    assert list(result[3].index) == [_NEW]


@pytest.mark.parametrize("defer", [True, False])
def test_repair_without_new_bar(defer):
    """新しい足が無く直す足だけの回でも落ちず、直した足で特徴量を作る"""
    true_bars = _true_bars()
    result, mock_save, base, _ = _run(_db_bars(), true_bars, defer, latest_only_overlap=True)

    assert list(_saved(result, mock_save, defer).index) == [_PARTIAL]
    assert result[2] is not None
    assert base.loc[_PARTIAL, "Close"] == true_bars.loc[_PARTIAL, "Close"]


def test_repair_does_not_bypass_stale_bars_guard():
    """足が止まった銘柄は、直す足があっても今までどおり特徴量を作らない"""
    with patch(f"{_P}._count_weekdays_behind", return_value=999):
        result, mock_save, base, _ = _run(_db_bars(), _true_bars(), latest_only_overlap=True)

    assert result is None
    assert base is None
    mock_save.assert_not_called()


def test_split_on_repaired_day_does_not_refetch():
    """重ねた日の Stock Splits では全期間を取り直さない"""
    fresh = _true_bars()
    fresh.loc[_PARTIAL, "Stock Splits"] = 2.0
    result, _, _, mock_get = _run(_db_bars(), fresh)

    assert mock_get.call_count == 1
    assert list(result[3].index) == [_PARTIAL, _NEW]


@pytest.mark.parametrize("cause", ["close", "split"])
def test_full_refetch_comes_first(cause):
    """終値が 3% を超えてずれた・新しい足に分割があるときは、今までどおり全期間を1回保存する"""
    db, fresh = _db_bars(), _true_bars()
    if cause == "close":
        db.loc[_PARTIAL, "Close"] *= 0.9
    else:
        fresh.loc[_NEW, "Stock Splits"] = 2.0
    result, mock_save, _, mock_get = _run(db, fresh, defer=False)

    assert mock_get.call_count == 2
    assert len(_saved(result, mock_save, False)) == len(fresh)


def test_batch_logs_repaired_count_and_dates_once():
    """直した銘柄の数と日付は、回ごとに1行だけ INFO で出す"""
    from src.market_data.split_refetch import record_bar_repairs

    def _fetch(market, symbol, **_kw):
        record_bar_repairs(market, symbol, _true_bars().loc[[_PARTIAL]])
        return None

    tasks = [{"market": "us", "symbol": s} for s in ("AAPL", "MU")]
    with (
        patch(f"{_P}.fetch_stock_data_with_features", side_effect=_fetch),
        patch("src.market_data.split_refetch.logger") as mock_logger,
    ):
        run_batch_pipeline(tasks, fetch_only=True)
        run_batch_pipeline([], fetch_only=True)

    assert mock_logger.info.call_count == 1
    message = mock_logger.info.call_args.args[0]
    assert "2銘柄" in message
    assert "2026-10-19: 2" in message
    mock_logger.warning.assert_not_called()


def test_save_raw_ohlcv_uses_same_key_as_db_bar():
    """取り直した足は、DB の足と同じ日時のキーで保存される（UPSERT で上書きになる）"""
    from src.market_data.saver import save_raw_ohlcv
    from src.market_data.split_refetch import partial_bar_repairs

    repairs = partial_bar_repairs(_db_bars(), _true_bars())
    with patch("src.market_data.saver.upsert_raw_ohlcv", return_value=1) as mock_upsert:
        save_raw_ohlcv("us", "AAPL", repairs)

    (row,) = mock_upsert.call_args.args[0]
    assert (row["market"], row["symbol"], row["timeframe"]) == ("us", "AAPL", "1d")
    assert row["ts"] == _PARTIAL
    assert row["volume"] == _true_bars().loc[_PARTIAL, "Volume"]
