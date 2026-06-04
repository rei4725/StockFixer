import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.screening.trend_screener import screen_trend_candidates

_N = 300  # 252日以上のデータを確保


def _make_df(closes, volume=1_000_000) -> pd.DataFrame:
    """Close 系列から stock_features 風 DataFrame を作る。"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": np.full(n, float(volume)),
        }
    )


def _uptrend(start=10.0, end=100.0, n=_N) -> pd.DataFrame:
    """単調上昇（200日線上・上向き・高値圏・高相対強度）。"""
    return _make_df(np.linspace(start, end, n))


def _downtrend(start=100.0, end=20.0, n=_N) -> pd.DataFrame:
    """単調下落（200日線割れ・高値から大きく下落）。"""
    return _make_df(np.linspace(start, end, n))


def _peaked(n=_N) -> pd.DataFrame:
    """前半上昇・後半下落で52週高値から大きく離れた銘柄。"""
    up = np.linspace(10.0, 100.0, n // 2)
    down = np.linspace(100.0, 50.0, n - n // 2)
    return _make_df(np.concatenate([up, down]))


def _patch_db(symbols, df_map):
    """get_all_symbols / load_stock_features をモックするコンテキスト。"""

    def _loader(market, symbol):
        return df_map.get((market, symbol))

    return (
        patch("src.screening.trend_screener.get_all_symbols", return_value=symbols),
        patch("src.screening.trend_screener.load_stock_features", side_effect=_loader),
    )


class TestScreenTrendCandidates(unittest.TestCase):
    def _run(self, symbols, df_map, **kwargs):
        p1, p2 = _patch_db(symbols, df_map)
        with p1, p2:
            return screen_trend_candidates(market="us", **kwargs)

    def test_uptrend_symbol_is_selected(self):
        """上昇トレンド銘柄が候補に入る。"""
        symbols = [("us", "UP"), ("us", "DOWN")]
        df_map = {("us", "UP"): _uptrend(), ("us", "DOWN"): _downtrend()}
        result = self._run(symbols, df_map, rel_strength_pct=0.0)
        codes = [c.symbol for c in result]
        self.assertIn("UP", codes)
        up = next(c for c in result if c.symbol == "UP")
        self.assertTrue(up.above_200dma)
        self.assertTrue(up.sma200_rising)

    def test_downtrend_symbol_is_excluded(self):
        """200日線割れ／高値から大きく下落した銘柄は除外される。"""
        symbols = [("us", "UP"), ("us", "DOWN"), ("us", "PEAK")]
        df_map = {
            ("us", "UP"): _uptrend(),
            ("us", "DOWN"): _downtrend(),
            ("us", "PEAK"): _peaked(),
        }
        result = self._run(symbols, df_map, rel_strength_pct=0.0)
        codes = [c.symbol for c in result]
        self.assertNotIn("DOWN", codes)
        self.assertNotIn("PEAK", codes)

    def test_insufficient_data_excluded(self):
        """min_data_days 未満の銘柄は除外される。"""
        symbols = [("us", "UP"), ("us", "SHORT")]
        df_map = {
            ("us", "UP"): _uptrend(),
            ("us", "SHORT"): _uptrend(n=100),  # 252日未満
        }
        result = self._run(symbols, df_map, rel_strength_pct=0.0)
        codes = [c.symbol for c in result]
        self.assertNotIn("SHORT", codes)
        self.assertIn("UP", codes)

    def test_low_liquidity_excluded(self):
        """低流動性銘柄は除外される。"""
        symbols = [("us", "UP"), ("us", "ILLIQUID")]
        df_map = {
            ("us", "UP"): _uptrend(),
            ("us", "ILLIQUID"): _make_df(np.linspace(10.0, 100.0, _N), volume=1_000),
        }
        result = self._run(symbols, df_map, rel_strength_pct=0.0, min_avg_volume=100_000)
        codes = [c.symbol for c in result]
        self.assertNotIn("ILLIQUID", codes)

    def test_low_price_excluded(self):
        """低価格銘柄は除外される。"""
        symbols = [("us", "UP"), ("us", "PENNY")]
        df_map = {
            ("us", "UP"): _uptrend(),
            ("us", "PENNY"): _make_df(np.linspace(0.5, 3.0, _N)),  # 最終 < 5.0
        }
        result = self._run(symbols, df_map, rel_strength_pct=0.0, min_price=5.0)
        codes = [c.symbol for c in result]
        self.assertNotIn("PENNY", codes)

    def test_top_n_limits_and_sorted_desc(self):
        """top_n で件数が切られ、スコア降順で並ぶ。"""
        symbols = [("us", f"S{i}") for i in range(6)]
        df_map = {("us", f"S{i}"): _uptrend(start=10.0, end=100.0 + i * 20) for i in range(6)}
        result = self._run(symbols, df_map, rel_strength_pct=0.0, top_n=3)
        self.assertEqual(len(result), 3)
        scores = [c.score for c in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_universe_returns_empty(self):
        """空データでも例外を投げず空リストを返す。"""
        result = self._run([], {})
        self.assertEqual(result, [])

    def test_all_excluded_returns_empty(self):
        """全銘柄が除外される場合も空リストを返す。"""
        symbols = [("us", "DOWN")]
        df_map = {("us", "DOWN"): _downtrend()}
        result = self._run(symbols, df_map, rel_strength_pct=0.0)
        self.assertEqual(result, [])

    def test_as_of_truncates_future_data(self):
        """as_of 指定時はその日以前のみで判定する（ルックアヘッド防止）。

        前半上昇・後半崩落の銘柄を、ピーク時点の as_of で評価すると上昇トレンドと
        して選ばれるが、崩落後の全データで評価すると除外される。
        """
        up = np.linspace(10.0, 100.0, 300)
        down = np.linspace(100.0, 30.0, 100)
        df = _make_df(np.concatenate([up, down]))
        symbols = [("us", "PEAKED")]
        df_map = {("us", "PEAKED"): df}

        peak_date = df["date"].iloc[299]
        result_as_of = self._run(symbols, df_map, rel_strength_pct=0.0, as_of=peak_date)
        self.assertIn("PEAKED", [c.symbol for c in result_as_of])

        # as_of なし（崩落後の全データ）では高値から大きく下落し除外される
        result_full = self._run(symbols, df_map, rel_strength_pct=0.0)
        self.assertNotIn("PEAKED", [c.symbol for c in result_full])


if __name__ == "__main__":
    unittest.main()
