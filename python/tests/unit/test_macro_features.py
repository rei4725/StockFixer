"""macro_features モジュールのユニットテスト"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.market_data.macro_features import (
    add_event_flags,
    add_macro_derived_features,
    fetch_additional_macro_features,
)


def _make_df(n: int = 30, include_vix: bool = True, include_tnx: bool = True) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data: dict = {"Close": np.linspace(100, 130, n).astype(float)}
    if include_vix:
        data["vix_close"] = np.linspace(15, 30, n).astype(float)
    if include_tnx:
        data["tnx_close"] = np.linspace(3.5, 4.5, n).astype(float)
    return pd.DataFrame(data, index=dates)


def _make_ticker_hist(close_val: float = 100.0, n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"Close": [close_val] * n}, index=dates)


class TestAddMacroDerivedFeatures(unittest.TestCase):
    """add_macro_derived_features() のテスト"""

    def test_vix_columns_added_when_vix_present(self):
        df = _make_df(30, include_vix=True, include_tnx=False)
        result = add_macro_derived_features(df)
        self.assertIn("vix_ma20", result.columns)
        self.assertIn("vix_spike", result.columns)
        self.assertIn("vix_momentum", result.columns)

    def test_tnx_momentum_added_when_tnx_present(self):
        df = _make_df(30, include_vix=False, include_tnx=True)
        result = add_macro_derived_features(df)
        self.assertIn("tnx_momentum", result.columns)

    def test_no_vix_columns_when_vix_absent(self):
        df = _make_df(30, include_vix=False, include_tnx=False)
        result = add_macro_derived_features(df)
        self.assertNotIn("vix_ma20", result.columns)
        self.assertNotIn("vix_spike", result.columns)
        self.assertNotIn("vix_momentum", result.columns)

    def test_no_tnx_momentum_when_tnx_absent(self):
        df = _make_df(30, include_vix=False, include_tnx=False)
        result = add_macro_derived_features(df)
        self.assertNotIn("tnx_momentum", result.columns)

    def test_vix_spike_is_binary(self):
        df = _make_df(30, include_vix=True)
        result = add_macro_derived_features(df)
        unique_vals = set(result["vix_spike"].unique())
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_vix_spike_threshold_at_25(self):
        """VIX が 25 超なら vix_spike=1、25 以下なら 0"""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"vix_close": [20.0, 25.0, 25.1, 30.0, 15.0]}, index=dates)
        result = add_macro_derived_features(df)
        self.assertEqual(list(result["vix_spike"]), [0, 0, 1, 1, 0])

    def test_vix_ma20_has_no_nan_due_to_min_periods(self):
        """min_periods=1 により先頭行でも NaN にならない"""
        df = _make_df(10, include_vix=True, include_tnx=False)
        result = add_macro_derived_features(df)
        self.assertEqual(result["vix_ma20"].isna().sum(), 0)

    def test_input_not_mutated(self):
        df = _make_df(30)
        orig_cols = set(df.columns)
        add_macro_derived_features(df)
        self.assertEqual(orig_cols, set(df.columns))

    def test_returns_dataframe(self):
        df = _make_df(30)
        result = add_macro_derived_features(df)
        self.assertIsInstance(result, pd.DataFrame)


class TestAddEventFlags(unittest.TestCase):
    """add_event_flags() のテスト"""

    def test_columns_added(self):
        df = _make_df(30)
        result = add_event_flags(df)
        self.assertIn("is_earnings_season", result.columns)
        self.assertIn("is_fomc_week", result.columns)

    def test_earnings_season_january_is_one(self):
        """1月は決算シーズン月なので is_earnings_season=1"""
        dates = pd.date_range("2024-01-10", periods=5, freq="D")
        df = pd.DataFrame({"Close": 100.0}, index=dates)
        result = add_event_flags(df)
        self.assertTrue((result["is_earnings_season"] == 1).all())

    def test_earnings_season_march_is_zero(self):
        """3月は決算シーズン外なので is_earnings_season=0"""
        dates = pd.date_range("2024-03-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": 100.0}, index=dates)
        result = add_event_flags(df)
        self.assertTrue((result["is_earnings_season"] == 0).all())

    def test_earnings_season_months_covered(self):
        """決算シーズン月（1,2,4,5,7,8,10,11）は 1、それ以外は 0"""
        earnings_months = {1, 2, 4, 5, 7, 8, 10, 11}
        non_earnings_months = {3, 6, 9, 12}
        for month in earnings_months:
            dates = pd.DatetimeIndex([pd.Timestamp(2024, month, 15)])
            df = pd.DataFrame({"Close": 100.0}, index=dates)
            result = add_event_flags(df)
            self.assertEqual(result["is_earnings_season"].iloc[0], 1, f"month={month} should be 1")
        for month in non_earnings_months:
            dates = pd.DatetimeIndex([pd.Timestamp(2024, month, 15)])
            df = pd.DataFrame({"Close": 100.0}, index=dates)
            result = add_event_flags(df)
            self.assertEqual(result["is_earnings_season"].iloc[0], 0, f"month={month} should be 0")

    def test_fomc_week_on_fomc_date_is_one(self):
        """FOMC 開催日当日は is_fomc_week=1（2025-01-29 はスケジュールに含まれる）"""
        dates = pd.DatetimeIndex([pd.Timestamp(2025, 1, 29)])
        df = pd.DataFrame({"Close": 100.0}, index=dates)
        result = add_event_flags(df)
        self.assertEqual(result["is_fomc_week"].iloc[0], 1)

    def test_fomc_week_within_3_days_is_one(self):
        """FOMC 開催日の ±3 日は is_fomc_week=1"""
        fomc_date = pd.Timestamp(2025, 1, 29)
        for delta in range(-3, 4):
            date = fomc_date + pd.Timedelta(days=delta)
            df = pd.DataFrame({"Close": 100.0}, index=pd.DatetimeIndex([date]))
            result = add_event_flags(df)
            self.assertEqual(result["is_fomc_week"].iloc[0], 1, f"delta={delta} should be 1")

    def test_fomc_week_4_days_away_is_zero(self):
        """FOMC 開催日の 4 日後は is_fomc_week=0（次の FOMC も考慮）"""
        # 2025-01-29 から 4 日後 = 2025-02-02、次の FOMC は 2025-03-19 で遠い
        dates = pd.DatetimeIndex([pd.Timestamp(2025, 2, 2)])
        df = pd.DataFrame({"Close": 100.0}, index=dates)
        result = add_event_flags(df)
        self.assertEqual(result["is_fomc_week"].iloc[0], 0)

    def test_non_datetimeindex_skips_gracefully(self):
        """DatetimeIndex でない場合はフラグ列を追加せずそのまま返す"""
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=[0, 1, 2])
        result = add_event_flags(df)
        self.assertNotIn("is_earnings_season", result.columns)
        self.assertNotIn("is_fomc_week", result.columns)

    def test_input_not_mutated(self):
        df = _make_df(30)
        orig_cols = set(df.columns)
        add_event_flags(df)
        self.assertEqual(orig_cols, set(df.columns))

    def test_returns_dataframe(self):
        df = _make_df(30)
        result = add_event_flags(df)
        self.assertIsInstance(result, pd.DataFrame)


class TestFetchAdditionalMacroFeatures(unittest.TestCase):
    """fetch_additional_macro_features() のテスト"""

    @patch("src.market_data.yf_client.ticker_history")
    def test_returns_dataframe_with_all_columns(self, mock_hist):
        """全ティッカー取得成功時に sp500_close, dxy_close, gold_close が返る"""
        mock_hist.return_value = _make_ticker_hist()
        result = fetch_additional_macro_features("2024-01-01", "2024-01-10")
        self.assertIsNotNone(result)
        for col in ("sp500_close", "dxy_close", "gold_close"):
            self.assertIn(col, result.columns)

    @patch("src.market_data.yf_client.ticker_history")
    def test_returns_none_when_all_fail(self, mock_hist):
        """全ティッカーの取得に失敗した場合は None を返す"""
        mock_hist.return_value = None
        result = fetch_additional_macro_features("2024-01-01", "2024-01-10")
        self.assertIsNone(result)

    @patch("src.market_data.yf_client.ticker_history")
    def test_partial_failure_returns_available_columns(self, mock_hist):
        """一部ティッカーの取得に失敗しても取得できたカラムが返る"""

        def side_effect(ticker, start, end):
            if ticker == "^GSPC":
                return _make_ticker_hist()
            return None

        mock_hist.side_effect = side_effect
        result = fetch_additional_macro_features("2024-01-01", "2024-01-10")
        self.assertIsNotNone(result)
        self.assertIn("sp500_close", result.columns)
        self.assertNotIn("dxy_close", result.columns)
        self.assertNotIn("gold_close", result.columns)

    @patch("src.market_data.yf_client.ticker_history")
    def test_index_is_tz_naive(self, mock_hist):
        """返却 DataFrame のインデックスがタイムゾーンなしであることを確認"""
        tz_index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        hist = pd.DataFrame({"Close": [100.0] * 5}, index=tz_index)
        mock_hist.return_value = hist
        result = fetch_additional_macro_features("2024-01-01", "2024-01-05")
        self.assertIsNotNone(result)
        self.assertIsNone(result.index.tz)

    @patch("src.market_data.yf_client.ticker_history")
    def test_exception_in_one_ticker_is_handled(self, mock_hist):
        """1つのティッカーで例外が発生しても他のティッカーは取得できる"""
        call_count = [0]

        def side_effect(ticker, start, end):
            call_count[0] += 1
            if ticker == "^GSPC":
                raise RuntimeError("network error")
            return _make_ticker_hist()

        mock_hist.side_effect = side_effect
        result = fetch_additional_macro_features("2024-01-01", "2024-01-10")
        self.assertIsNotNone(result)
        self.assertNotIn("sp500_close", result.columns)
        self.assertIn("dxy_close", result.columns)


class TestMacroFeaturesIntegration(unittest.TestCase):
    """マクロ特徴量が pipeline の DataFrame に組み込まれることを確認する結合テスト"""

    def test_derived_features_on_cross_asset_data(self):
        """VIX・TNX 列を含む DataFrame に派生マクロ指標が追加される"""
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "Close": np.linspace(100, 130, n),
                "vix_close": np.linspace(18, 32, n),
                "tnx_close": np.linspace(3.8, 4.5, n),
            },
            index=dates,
        )
        result = add_macro_derived_features(df)
        result = add_event_flags(result)

        # 派生マクロ指標
        for col in ("vix_ma20", "vix_spike", "vix_momentum", "tnx_momentum"):
            self.assertIn(col, result.columns, f"{col} missing from result")
        # イベントフラグ
        for col in ("is_earnings_season", "is_fomc_week"):
            self.assertIn(col, result.columns, f"{col} missing from result")
        # NaN がないこと（min_periods=1 で補完済みのはず）
        self.assertEqual(result["vix_ma20"].isna().sum(), 0)
        self.assertEqual(result["is_earnings_season"].isna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
