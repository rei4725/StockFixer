"""クロスアセットと追加マクロの終値の取得のテスト（#1988）

2つの関数が同じ ticker_history の偽物に対して同じ表か None を返すこと、
取れなかったティッカーの例外が WARNING で1行残ることを確かめる。
"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.market_data.loader import fetch_cross_asset_features
from src.market_data.macro_features import fetch_additional_macro_features

_HIST = "src.market_data.yf_client.ticker_history"
_LOGGER = "src.market_data.macro_features"

_CROSS = {"^VIX": "vix_close", "JPY=X": "usdjpy_close", "^TNX": "tnx_close"}
_ADDITIONAL = {"^GSPC": "sp500_close", "DX-Y.NYB": "dxy_close", "GC=F": "gold_close"}


def _hist(base: float, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"Close": [base + i for i in range(len(dates))]}, index=dates)


def _multi_close(base: float, dates: pd.DatetimeIndex) -> pd.DataFrame:
    cols = pd.MultiIndex.from_tuples([("Close", "X"), ("Close", "Y"), ("Open", "X")])
    rows = [[base + i, -1.0, base - 1 + i] for i in range(len(dates))]
    return pd.DataFrame(rows, index=dates, columns=cols)


_NAIVE = pd.date_range("2024-01-03", periods=3, freq="D")
_TZ = pd.date_range("2024-01-01", periods=4, freq="D", tz="America/New_York")


def _fake(order: list[str]):
    """1つ目: 成功（tz 付き）、2つ目: 多段の Close、3つ目: 例外。"""

    def side_effect(ticker, start, end):
        if ticker == order[0]:
            return _hist(10.0, _TZ)
        if ticker == order[1]:
            return _multi_close(50.0, _NAIVE)
        raise RuntimeError(f"boom {ticker}")

    return side_effect


def _expected(first: str, second: str) -> pd.DataFrame:
    a = pd.Series([10.0, 11.0, 12.0, 13.0], index=_TZ.tz_localize(None), name=first)
    b = pd.Series([50.0, 51.0, 52.0], index=_NAIVE, name=second)
    out = pd.concat([a, b], axis=1, sort=False)
    out.sort_index(inplace=True)
    return out


class TestSameFrames(unittest.TestCase):
    """直す前と同じ表を返すこと"""

    @patch(_HIST)
    def test_cross_asset_mixed(self, mock_hist):
        mock_hist.side_effect = _fake(["^VIX", "JPY=X"])
        result = fetch_cross_asset_features("2024-01-01", "2024-01-10")
        pd.testing.assert_frame_equal(result, _expected("vix_close", "usdjpy_close"))

    @patch(_HIST)
    def test_additional_mixed(self, mock_hist):
        mock_hist.side_effect = _fake(["^GSPC", "DX-Y.NYB"])
        result = fetch_additional_macro_features("2024-01-01", "2024-01-10")
        pd.testing.assert_frame_equal(result, _expected("sp500_close", "dxy_close"))

    @patch(_HIST)
    def test_end_date_is_extended_by_one_day(self, mock_hist):
        mock_hist.return_value = _hist(1.0, _NAIVE)
        fetch_cross_asset_features("2024-01-01", "2024-01-10")
        fetch_additional_macro_features("2024-01-01", "2024-01-10")
        tickers = [c.args[0] for c in mock_hist.call_args_list]
        self.assertEqual(tickers, list(_CROSS) + list(_ADDITIONAL))
        for c in mock_hist.call_args_list:
            self.assertEqual(c.kwargs, {"start": "2024-01-01", "end": "2024-01-11"})

    @patch(_HIST)
    def test_all_success_column_order(self, mock_hist):
        mock_hist.return_value = _hist(1.0, _NAIVE)
        cross = fetch_cross_asset_features("2024-01-01", "2024-01-10")
        additional = fetch_additional_macro_features("2024-01-01", "2024-01-10")
        self.assertEqual(list(cross.columns), list(_CROSS.values()))
        self.assertEqual(list(additional.columns), list(_ADDITIONAL.values()))

    @patch(_HIST)
    def test_none_empty_and_no_close_give_none(self, mock_hist):
        for value in (None, pd.DataFrame(), pd.DataFrame({"Open": [1.0]}, index=_NAIVE[:1])):
            mock_hist.side_effect = None
            mock_hist.return_value = value
            self.assertIsNone(fetch_cross_asset_features("2024-01-01", "2024-01-10"))
            self.assertIsNone(fetch_additional_macro_features("2024-01-01", "2024-01-10"))

    @patch(_HIST)
    def test_all_exceptions_give_none(self, mock_hist):
        mock_hist.side_effect = RuntimeError("down")
        self.assertIsNone(fetch_cross_asset_features("2024-01-01", "2024-01-10"))
        self.assertIsNone(fetch_additional_macro_features("2024-01-01", "2024-01-10"))


class TestWarningOnException(unittest.TestCase):
    """取れなかったティッカーの例外が WARNING で残ること"""

    @patch(_HIST)
    def test_cross_asset_warns_with_ticker_and_error(self, mock_hist):
        mock_hist.side_effect = _fake(["^VIX", "JPY=X"])
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            fetch_cross_asset_features("2024-01-01", "2024-01-10")
        warnings = [r for r in cm.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        msg = warnings[0].getMessage()
        self.assertIn("^TNX", msg)
        self.assertIn("RuntimeError", msg)
        self.assertIn("boom ^TNX", msg)

    @patch(_HIST)
    def test_additional_warns_with_ticker_and_error(self, mock_hist):
        mock_hist.side_effect = _fake(["^GSPC", "DX-Y.NYB"])
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            fetch_additional_macro_features("2024-01-01", "2024-01-10")
        warnings = [r for r in cm.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        msg = warnings[0].getMessage()
        self.assertIn("GC=F", msg)
        self.assertIn("RuntimeError", msg)
        self.assertIn("boom GC=F", msg)


if __name__ == "__main__":
    unittest.main()
