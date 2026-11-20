"""相場の指標の結合と欠けの判定のテスト（#2057）

join_macro_frame が前方補完だけで埋め、先頭の NaN を後ろの値で埋めない（未来の値を入れない）ことを留める。
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.market_data.macro_join import (
    _ADDITIONAL_MACRO_TICKERS,
    _CROSS_ASSET_TICKERS,
    join_macro_frame,
    missing_macro_columns,
)

_CROSS_COLS = list(_CROSS_ASSET_TICKERS)
_ADDITIONAL_COLS = list(_ADDITIONAL_MACRO_TICKERS)


def _bars(tz: str | None = None) -> pd.DataFrame:
    dates = pd.bdate_range("2026-10-12", periods=5, tz=tz)
    return pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)


def _macro(values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-10-12", periods=len(values))
    return pd.DataFrame({_CROSS_COLS[0]: values}, index=dates)


def _full(cols: list[str]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-10-12", periods=3)
    return pd.DataFrame({col: [1.0, 2.0, 3.0] for col in cols}, index=dates)


def _assert_leading_nan_kept() -> None:
    col = _CROSS_COLS[0]
    out = join_macro_frame(
        _bars(), _macro([np.nan, np.nan, 20.0, 21.0, 22.0]), "テスト", "us", "AAPL"
    )
    assert out[col].iloc[0:2].isna().all()
    assert out[col].iloc[2:].tolist() == [20.0, 21.0, 22.0]


class TestJoinMacroFrame:
    def test_leading_nan_stays_nan(self) -> None:
        """先頭の欠けは後ろの日の値で埋めない（bfill の未来情報リークを止める）。"""
        _assert_leading_nan_kept()

    def test_leading_nan_check_catches_bfill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ffill を ffill().bfill() に替えると、先頭の NaN の確かめが落ちる。"""
        original_ffill = pd.Series.ffill

        def ffill_then_bfill(self: pd.Series, *args: Any, **kwargs: Any) -> pd.Series:
            return original_ffill(self, *args, **kwargs).bfill()

        monkeypatch.setattr(pd.Series, "ffill", ffill_then_bfill)
        with pytest.raises(AssertionError):
            _assert_leading_nan_kept()

    def test_middle_gap_is_filled_with_previous_day(self) -> None:
        col = _CROSS_COLS[0]
        out = join_macro_frame(
            _bars(), _macro([10.0, np.nan, np.nan, 13.0, 14.0]), "テスト", "us", "AAPL"
        )
        assert out[col].tolist() == [10.0, 10.0, 10.0, 13.0, 14.0]

    def test_dates_missing_in_frame_are_filled_with_previous_day(self) -> None:
        col = _CROSS_COLS[0]
        frame = _macro([10.0, 11.0, 12.0, 13.0, 14.0]).drop(pd.Timestamp("2026-10-14"))
        out = join_macro_frame(_bars(), frame, "テスト", "us", "AAPL")
        assert out[col].tolist() == [10.0, 11.0, 11.0, 13.0, 14.0]

    def test_tz_aware_index_is_joined(self) -> None:
        col = _CROSS_COLS[0]
        out = join_macro_frame(
            _bars(tz="UTC"), _macro([10.0, 11.0, 12.0, 13.0, 14.0]), "テスト", "us", "AAPL"
        )
        assert isinstance(out.index, pd.DatetimeIndex)
        assert out.index.tz is None
        assert out[col].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
        assert out["Close"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]

    def test_none_frame_returns_df_unchanged(self) -> None:
        df = _bars()
        out = join_macro_frame(df, None, "テスト", "us", "AAPL")
        assert out is df
        assert list(out.columns) == ["Close"]

    def test_empty_frame_returns_df_unchanged(self) -> None:
        df = _bars()
        out = join_macro_frame(df, pd.DataFrame(), "テスト", "us", "AAPL")
        assert out is df
        assert list(out.columns) == ["Close"]


class TestMissingMacroColumns:
    def test_both_complete_returns_empty(self) -> None:
        assert missing_macro_columns(_full(_CROSS_COLS), _full(_ADDITIONAL_COLS)) == []

    def test_cross_asset_none_returns_all_its_columns(self) -> None:
        assert missing_macro_columns(None, _full(_ADDITIONAL_COLS)) == _CROSS_COLS

    def test_cross_asset_empty_returns_all_its_columns(self) -> None:
        assert missing_macro_columns(pd.DataFrame(), _full(_ADDITIONAL_COLS)) == _CROSS_COLS

    def test_additional_none_returns_all_its_columns(self) -> None:
        assert missing_macro_columns(_full(_CROSS_COLS), None) == _ADDITIONAL_COLS

    def test_additional_empty_returns_all_its_columns(self) -> None:
        assert missing_macro_columns(_full(_CROSS_COLS), pd.DataFrame()) == _ADDITIONAL_COLS

    def test_both_none_returns_every_column(self) -> None:
        assert missing_macro_columns(None, None) == _CROSS_COLS + _ADDITIONAL_COLS

    def test_partial_columns_returns_only_absent_ones(self) -> None:
        cross = _full(_CROSS_COLS[1:])
        additional = _full(_ADDITIONAL_COLS[:-1])
        assert missing_macro_columns(cross, additional) == [
            _CROSS_COLS[0],
            _ADDITIONAL_COLS[-1],
        ]
