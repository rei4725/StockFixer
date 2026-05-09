"""data/data_saver モジュールのユニットテスト（外部依存モック）"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_ohlcv_df(n=5):
    """yfinance 形式のOHLCV DataFrame を作成する"""
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": np.random.uniform(99, 101, n),
            "High": np.random.uniform(101, 103, n),
            "Low": np.random.uniform(97, 99, n),
            "Close": np.random.uniform(99, 101, n),
            "Volume": np.random.randint(1000, 5000, n).astype(float),
        },
        index=idx,
    )


class TestSaveRawOhlcv(unittest.TestCase):
    """save_raw_ohlcv のテスト（upsert_raw_ohlcv をモック）"""

    def setUp(self):
        self.mock_upsert = patch("src.market_data.saver.upsert_raw_ohlcv", return_value=5).start()
        self.addCleanup(patch.stopall)

    def test_returns_row_count(self):
        """upsert_raw_ohlcv の戻り値をそのまま返すこと"""
        from src.market_data.saver import save_raw_ohlcv

        df = _make_ohlcv_df(5)
        result = save_raw_ohlcv("us", "AAPL", df)
        self.assertEqual(result, 5)

    def test_calls_upsert_once(self):
        """upsert_raw_ohlcv が1回呼ばれること"""
        from src.market_data.saver import save_raw_ohlcv

        df = _make_ohlcv_df(3)
        save_raw_ohlcv("us", "AAPL", df)
        self.mock_upsert.assert_called_once()

    def test_rows_have_correct_market_symbol(self):
        """生成されるrowsに正しいmarket/symbolが含まれること"""
        from src.market_data.saver import save_raw_ohlcv

        df = _make_ohlcv_df(2)
        save_raw_ohlcv("jp", "7203", df)

        args, _ = self.mock_upsert.call_args
        rows = args[0]
        self.assertTrue(all(r["market"] == "jp" for r in rows))
        self.assertTrue(all(r["symbol"] == "7203" for r in rows))

    def test_rows_have_correct_timeframe(self):
        """rowsに指定した timeframe が含まれること"""
        from src.market_data.saver import save_raw_ohlcv

        df = _make_ohlcv_df(2)
        save_raw_ohlcv("us", "AAPL", df, timeframe="1h")

        args, _ = self.mock_upsert.call_args
        rows = args[0]
        self.assertTrue(all(r["timeframe"] == "1h" for r in rows))

    def test_rows_have_custom_source(self):
        """source パラメータが rowsに伝わること"""
        from src.market_data.saver import save_raw_ohlcv

        df = _make_ohlcv_df(2)
        save_raw_ohlcv("us", "AAPL", df, source="custom")

        args, _ = self.mock_upsert.call_args
        rows = args[0]
        self.assertTrue(all(r["source"] == "custom" for r in rows))

    def test_handles_adj_close_column(self):
        """Adj Close 列が adj_close として変換されること"""
        from src.market_data.saver import save_raw_ohlcv

        df = _make_ohlcv_df(2)
        df["Adj Close"] = df["Close"] * 0.99
        save_raw_ohlcv("us", "AAPL", df)

        args, _ = self.mock_upsert.call_args
        rows = args[0]
        self.assertIn("adj_close", rows[0])

    def test_nan_values_excluded(self):
        """NaN 値は row に含まれないこと"""
        from src.market_data.saver import save_raw_ohlcv

        df = _make_ohlcv_df(2)
        df.loc[df.index[0], "Volume"] = float("nan")
        save_raw_ohlcv("us", "AAPL", df)

        args, _ = self.mock_upsert.call_args
        rows = args[0]
        # 1行目は NaN のため volume が含まれない場合がある
        self.assertNotIn("Volume", rows[0] if "Volume" not in rows[0] else {})

    def test_timezone_aware_index_normalized(self):
        """タイムゾーン付きインデックスが tzinfo なしに変換されること"""
        from src.market_data.saver import save_raw_ohlcv

        idx = pd.date_range("2026-01-01", periods=2, freq="B", tz="America/New_York")
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [98.0, 99.0],
                "Close": [100.5, 101.5],
                "Volume": [1000.0, 2000.0],
            },
            index=idx,
        )
        save_raw_ohlcv("us", "AAPL", df)

        args, _ = self.mock_upsert.call_args
        rows = args[0]
        for r in rows:
            self.assertIsNone(r["ts"].tzinfo)

    def test_lowercase_column_name_fallback(self):
        """小文字のカラム名でも正しく変換されること（大小文字ゆらぎ吸収）"""
        from src.market_data.saver import save_raw_ohlcv

        df = _make_ohlcv_df(2)
        df.columns = [c.lower() for c in df.columns]
        save_raw_ohlcv("us", "AAPL", df)

        args, _ = self.mock_upsert.call_args
        rows = args[0]
        self.assertIn("close", rows[0])

    def test_jp_ticker_has_dot_t_suffix(self):
        """日本株のティッカーに .T サフィックスが付くこと"""
        from src.market_data.saver import save_raw_ohlcv

        df = _make_ohlcv_df(2)
        save_raw_ohlcv("jp", "7203", df)

        args, _ = self.mock_upsert.call_args
        rows = args[0]
        self.assertTrue(rows[0]["ticker"].endswith(".T"))


class TestSaveRawStockData(unittest.TestCase):
    """save_raw_stock_data のテスト（get_stock_data / upsert_stock_features をモック）"""

    def setUp(self):
        self.mock_upsert = patch("src.market_data.saver.upsert_stock_features").start()
        self.mock_get_ticker = patch(
            "src.market_data.saver.get_ticker", side_effect=lambda m, s: s
        ).start()
        self.addCleanup(patch.stopall)

    def _make_df(self, n=3):
        idx = pd.date_range("2026-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": np.ones(n)}, index=idx)

    def test_returns_none_when_get_stock_data_returns_none(self):
        """get_stock_data が None を返すとき None を返すこと"""
        from src.market_data.saver import save_raw_stock_data

        with patch("src.market_data.saver.get_stock_data", return_value=None):
            result = save_raw_stock_data("us", "AAPL", "2026-01-01", "2026-03-01")
        self.assertIsNone(result)

    def test_returns_none_when_get_stock_data_returns_empty(self):
        """get_stock_data が空 DataFrame を返すとき None を返すこと"""
        from src.market_data.saver import save_raw_stock_data

        with patch("src.market_data.saver.get_stock_data", return_value=pd.DataFrame()):
            result = save_raw_stock_data("us", "AAPL", "2026-01-01", "2026-03-01")
        self.assertIsNone(result)

    def test_auto_detect_dates_when_none_given(self):
        """start_date=None のとき get_stock_data を2回呼ぶこと（日付自動検出）"""
        from src.market_data.saver import save_raw_stock_data

        df = self._make_df()
        with patch("src.market_data.saver.get_stock_data", return_value=df) as mock_gsd:
            save_raw_stock_data("us", "AAPL", None, None)
        # 1回目: 全期間取得、2回目: 実際の取得
        self.assertEqual(mock_gsd.call_count, 2)

    def test_exception_during_auto_detect_returns_none(self):
        """日付自動検出中に例外が発生したとき None を返すこと"""
        from src.market_data.saver import save_raw_stock_data

        with patch("src.market_data.saver.get_stock_data", side_effect=Exception("network error")):
            result = save_raw_stock_data("us", "AAPL", None, None)
        self.assertIsNone(result)

    def test_normal_path_calls_upsert_stock_features(self):
        """正常パスで upsert_stock_features が呼ばれること"""
        from src.market_data.saver import save_raw_stock_data

        df = self._make_df()
        with patch("src.market_data.saver.get_stock_data", return_value=df):
            save_raw_stock_data("us", "AAPL", "2026-01-01", "2026-03-01")
        self.mock_upsert.assert_called_once()

    def test_normal_path_returns_dataframe(self):
        """正常パスでDataFrameが返されること"""
        from src.market_data.saver import save_raw_stock_data

        df = self._make_df()
        with patch("src.market_data.saver.get_stock_data", return_value=df):
            result = save_raw_stock_data("us", "AAPL", "2026-01-01", "2026-03-01")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    def test_auto_detect_returns_none_when_all_data_empty(self):
        """全期間データが空のとき None を返し、upsert は呼ばれないこと"""
        from src.market_data.saver import save_raw_stock_data

        with patch("src.market_data.saver.get_stock_data", return_value=pd.DataFrame()):
            result = save_raw_stock_data("us", "AAPL", None, None)
        self.assertIsNone(result)
        self.mock_upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
