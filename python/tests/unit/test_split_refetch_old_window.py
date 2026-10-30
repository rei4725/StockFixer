"""分割の取り直しで、取得の窓より前の DB の足に分割の比の段差を残さないテスト（#1392）"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.market_data.pipeline import fetch_stock_data_with_features
from src.market_data.split_refetch import find_price_steps

_P = "src.market_data.pipeline"
_PRICE_COLS = ("Open", "High", "Low", "Close")
# 取得の窓（既定の5年の窓の代わり）
_WINDOW_START = "2023-01-01"
_END = "2023-03-07"


def _make_ohlcv() -> pd.DataFrame:
    """窓より前の 2022-10-01 から窓の終わりまでの、分割で調整し直した足"""
    dates = pd.date_range("2022-10-01", _END, freq="D")
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 120, n),
            "High": np.linspace(101, 121, n),
            "Low": np.linspace(99, 119, n),
            "Close": np.linspace(100, 120, n),
            "Volume": np.full(n, 3000),
        },
        index=dates,
    )


class _FakeRawStore:
    """market_data_raw の差し替え。(ts) の UPSERT と期間の読み出しだけを持つ"""

    def __init__(self, bars: pd.DataFrame):
        self.bars = bars.copy()

    def load(self, _market, _symbol, start_date=None, end_date=None, _timeframe="1d"):
        out = self.bars
        if start_date is not None:
            out = out[out.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            out = out[out.index <= pd.Timestamp(end_date)]
        return None if out.empty else out.copy()

    def save(self, _market, _symbol, df: pd.DataFrame) -> None:
        cols = list(self.bars.columns)
        merged = pd.concat([self.bars, df[cols]])
        self.bars = merged[~merged.index.duplicated(keep="last")].sort_index()


def _yf(bars: pd.DataFrame):
    def _fetch(_market, _ticker, start, end):
        return bars.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()

    return _fetch


def _with_actions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Dividends"] = 0.0
    out["Stock Splits"] = 0.0
    return out


@patch(f"{_P}.fetch_pageview_features", return_value=None)
@patch(f"{_P}.fetch_news_sentiment_with_llm", return_value=None)
@patch(f"{_P}.fetch_additional_macro_features", return_value=None)
@patch(f"{_P}.fetch_cross_asset_features", return_value=None)
@patch(f"{_P}.should_fetch_fresh_data", return_value=True)
class TestSplitRefetchOldWindow(unittest.TestCase):
    def _run(self, store: _FakeRawStore, yf_bars: pd.DataFrame, symbol: str):
        with (
            patch(f"{_P}.get_raw_ohlcv_from_db", side_effect=store.load),
            patch(f"{_P}.save_raw_ohlcv", side_effect=store.save),
            patch(f"{_P}.get_stock_data", side_effect=_yf(yf_bars)) as mock_get,
        ):
            result = fetch_stock_data_with_features(
                market="jp", symbol=symbol, start_date=_WINDOW_START, end_date=_END
            )
        return result, mock_get

    def test_split_refetch_leaves_no_step_before_window(self, *_mocks):
        """窓より前に分割前の足を持つ銘柄で取り直すと、全期間の足に段差が残らない"""
        adjusted = _make_ohlcv()
        # DB の足は 2023-03-01 まで。1:5 の分割は 2023-03-03 で、DB の足はすべて分割前の値段
        db_bars = adjusted.loc[:"2023-03-01"].copy()
        for col in _PRICE_COLS:
            db_bars[col] *= 5.0
        yf_bars = _with_actions(adjusted)
        yf_bars.loc["2023-03-03", "Stock Splits"] = 5.0
        store = _FakeRawStore(db_bars)

        result, _mock_get = self._run(store, yf_bars, "8035")

        self.assertIsNotNone(result)
        full = store.load("jp", "8035")
        self.assertEqual(find_price_steps(full), [])
        self.assertEqual(full.index.min(), pd.Timestamp("2022-10-01"))
        self.assertAlmostEqual(full["Close"].iloc[0], adjusted["Close"].iloc[0])
        # 特徴量は今までどおり窓の中の足で作る
        _m, _s, data, _raw, _q = result
        self.assertGreaterEqual(data.index.min(), pd.Timestamp(_WINDOW_START))

    def test_daily_diff_without_split_keeps_rows_before_window(self, *_mocks):
        """分割の無い日次の差分取得では、窓より前の行に触らない"""
        adjusted = _make_ohlcv()
        db_bars = adjusted.loc[:"2023-03-01"].copy()
        # 窓より前の行に印をつけ、上書きされないことを確かめる
        db_bars.loc[:"2022-12-31", "Volume"] = 1234
        store = _FakeRawStore(db_bars)
        before = store.bars.loc[:"2022-12-31"].copy()

        result, mock_get = self._run(store, _with_actions(adjusted), "7203")

        self.assertIsNotNone(result)
        mock_get.assert_called_once()
        self.assertEqual(mock_get.call_args.args[2], "2023-02-25")
        pd.testing.assert_frame_equal(store.bars.loc[:"2022-12-31"], before)
        self.assertEqual(store.bars.index.max(), pd.Timestamp(_END))


if __name__ == "__main__":
    unittest.main()
