import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.screening.trend_screener import (
    ScreeningDataError,
    _compute_metrics,
    screen_trend_candidates,
)

_N = 300  # 252日以上のデータを確保
_TODAY = date(2026, 11, 4)  # 水曜。_make_df の足は既定でこの日に終わる


def _make_df(closes, volume=1_000_000, last_day=_TODAY) -> pd.DataFrame:
    """Close 系列から load_raw_ohlcv 風（Date index）DataFrame を作る。"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    idx = pd.date_range(end=last_day, periods=n, freq="B")
    idx.name = "Date"
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": np.full(n, float(volume)),
        },
        index=idx,
    )


def _uptrend(start=10.0, end=100.0, n=_N, last_day=_TODAY) -> pd.DataFrame:
    """単調上昇（200日線上・上向き・高値圏・高相対強度）。"""
    return _make_df(np.linspace(start, end, n), last_day=last_day)


def _downtrend(start=100.0, end=20.0, n=_N) -> pd.DataFrame:
    """単調下落（200日線割れ・高値から大きく下落）。"""
    return _make_df(np.linspace(start, end, n))


def _peaked(n=_N) -> pd.DataFrame:
    """前半上昇・後半下落で52週高値から大きく離れた銘柄。"""
    up = np.linspace(10.0, 100.0, n // 2)
    down = np.linspace(100.0, 50.0, n - n // 2)
    return _make_df(np.concatenate([up, down]))


def _patch_db(symbols, df_map):
    """load_all_raw_ohlcv_symbols / load_raw_ohlcv をモックするコンテキスト。"""

    def _loader(market, symbol):
        return df_map.get((market, symbol))

    return (
        patch("src.screening.trend_screener.load_all_raw_ohlcv_symbols", return_value=symbols),
        patch("src.screening.trend_screener.load_raw_ohlcv", side_effect=_loader),
    )


def _screen(symbols, df_map, today=_TODAY, **kwargs):
    """DB と今日の日付をモックしてスクリーンを流す。"""
    p1, p2 = _patch_db(symbols, df_map)
    with p1, p2, patch("src.screening.trend_screener._today", return_value=today):
        return screen_trend_candidates(market="us", **kwargs)


class TestScreenTrendCandidates(unittest.TestCase):
    def _run(self, symbols, df_map, **kwargs):
        return _screen(symbols, df_map, **kwargs)

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

    def test_empty_universe_raises(self):
        """対象が0銘柄の回は「該当なし」と分けて例外にする（#1596）。"""
        with self.assertRaises(ScreeningDataError):
            self._run([], {})

    def test_no_symbol_for_market_raises(self):
        """別の市場の銘柄しか無い回も対象0銘柄として例外にする。"""
        with self.assertRaises(ScreeningDataError):
            self._run([("jp", "7203")], {("jp", "7203"): _uptrend()})

    def test_all_prices_unreadable_raises(self):
        """全銘柄の値段が読めない回は「該当なし」と分けて例外にする（#1596）。"""
        with self.assertRaises(ScreeningDataError):
            self._run([("us", "A"), ("us", "B")], {})

    def test_some_prices_unreadable_keeps_the_rest(self):
        """一部の銘柄が読めない回は残りの銘柄で判定する。"""
        symbols = [("us", "UP"), ("us", "GONE")]
        result = self._run(symbols, {("us", "UP"): _uptrend()}, rel_strength_pct=0.0)
        self.assertEqual([c.symbol for c in result], ["UP"])

    def test_as_of_before_all_data_returns_empty(self):
        """値段が読めて as_of より前の行が無いだけの回は、例外にせず該当なしとする。"""
        symbols = [("us", "UP")]
        result = self._run(symbols, {("us", "UP"): _uptrend()}, as_of="2000-01-01")
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

        peak_date = df.index[299].strftime("%Y-%m-%d")
        result_as_of = self._run(symbols, df_map, rel_strength_pct=0.0, as_of=peak_date)
        self.assertIn("PEAKED", [c.symbol for c in result_as_of])

        # as_of なし（崩落後の全データ）では高値から大きく下落し除外される
        result_full = self._run(symbols, df_map, rel_strength_pct=0.0)
        self.assertNotIn("PEAKED", [c.symbol for c in result_full])

    def test_nan_close_mid_series_keeps_candidate(self):
        """途中に終値 NaN の足が1本あっても候補に残る（#1597）。"""
        closes = np.linspace(10.0, 100.0, _N)
        closes[-100] = np.nan
        symbols = [("us", "GAP")]
        result = self._run(symbols, {("us", "GAP"): _make_df(closes)}, rel_strength_pct=0.0)
        self.assertEqual([c.symbol for c in result], ["GAP"])

    def test_too_few_valid_closes_excluded(self):
        """行が252本あっても有効な足が足りない銘柄は候補から外れる（#1597）。"""
        closes = np.linspace(10.0, 100.0, 252)
        closes[100] = np.nan
        symbols = [("us", "UP"), ("us", "THIN")]
        df_map = {("us", "UP"): _uptrend(), ("us", "THIN"): _make_df(closes)}
        result = self._run(symbols, df_map, rel_strength_pct=0.0)
        self.assertEqual([c.symbol for c in result], ["UP"])


class TestStaleSymbols(unittest.TestCase):
    """最後の足が基準の日より線を超えて古い銘柄を母集団から外す（#1635）。"""

    def _codes(self, df_map, **kwargs):
        result = _screen(list(df_map), df_map, rel_strength_pct=0.0, **kwargs)
        return [c.symbol for c in result]

    def test_stale_symbol_is_dropped_and_counted_in_log(self):
        """営業日で4日古い銘柄は候補に出ず、外した数がログに出る。"""
        df_map = {
            ("us", "UP"): _uptrend(),
            ("us", "STOP"): _uptrend(end=300.0, last_day=date(2026, 10, 29)),
        }
        with self.assertLogs("src.screening.trend_screener", level="WARNING") as logs:
            codes = self._codes(df_map)
        self.assertEqual(codes, ["UP"])
        output = " ".join(logs.output)
        self.assertIn("外した: 1/2 銘柄", output)
        self.assertIn("STOP", output)

    def test_symbol_at_the_line_is_kept(self):
        """営業日で3日古いだけの銘柄は外れない。"""
        df_map = {("us", "LATE"): _uptrend(last_day=date(2026, 10, 30))}
        self.assertEqual(self._codes(df_map), ["LATE"])

    def test_weekend_is_not_counted(self):
        """週末は数えない。日曜の回で火曜の足は3営業日の遅れである。"""
        df_map = {("us", "LATE"): _uptrend(last_day=date(2026, 11, 3))}
        self.assertEqual(self._codes(df_map, today=date(2026, 11, 8)), ["LATE"])

    def test_us_holiday_is_not_counted(self):
        """感謝祭をまたいだだけなら外れない（平日で数えると4日になる）。"""
        df_map = {("us", "LATE"): _uptrend(last_day=date(2026, 11, 24))}
        self.assertEqual(self._codes(df_map, today=date(2026, 11, 30)), ["LATE"])

    def test_weekend_and_us_holiday_are_not_counted(self):
        """週末と労働者の日を続けてまたいでも外れない（平日で数えると4日になる）。"""
        df_map = {("us", "LATE"): _uptrend(last_day=date(2026, 9, 2))}
        self.assertEqual(self._codes(df_map, today=date(2026, 9, 8)), ["LATE"])

    def test_last_valid_close_decides_staleness(self):
        """止まった後の足の終値が NaN なら、その足を新しい足として数えない。"""
        closes = np.linspace(10.0, 100.0, _N)
        closes[-4:] = np.nan
        df_map = {("us", "UP"): _uptrend(), ("us", "NANTAIL"): _make_df(closes)}
        self.assertEqual(self._codes(df_map), ["UP"])

    def test_as_of_is_the_base_day(self):
        """as_of を渡すと、今日ではなく as_of を基準に数える。"""
        as_of = date(2026, 6, 1)
        df_map = {
            ("us", "LONG"): _uptrend(n=_N + 200),  # as_of より後の足は切り詰められる
            ("us", "ENDS"): _uptrend(last_day=as_of),
            ("us", "STOP"): _uptrend(last_day=date(2026, 5, 26)),
        }
        codes = self._codes(df_map, as_of=as_of.isoformat())
        self.assertEqual(sorted(codes), ["ENDS", "LONG"])

    def test_as_of_with_all_symbols_stale_returns_empty(self):
        """as_of を渡す回は全銘柄が線より古くても例外にせず、該当なしとする。"""
        df_map = {("us", "STOP"): _uptrend(last_day=date(2026, 5, 26))}
        self.assertEqual(self._codes(df_map, as_of="2026-06-01"), [])

    def test_all_symbols_stale_raises(self):
        """as_of 無しで全銘柄が線より古い回は、成功にせず例外にする。"""
        df_map = {
            ("us", "A"): _uptrend(last_day=date(2026, 10, 1)),
            ("us", "B"): _uptrend(last_day=date(2026, 10, 2)),
        }
        with self.assertRaises(ScreeningDataError):
            self._codes(df_map)

    def test_all_readable_symbols_stale_raises(self):
        """読めない銘柄があっても、読めた銘柄の全部が線より古ければ例外にする。"""
        symbols = [("us", "STOP"), ("us", "GONE")]
        df_map = {("us", "STOP"): _uptrend(last_day=date(2026, 10, 1))}
        with self.assertRaises(ScreeningDataError):
            _screen(symbols, df_map, rel_strength_pct=0.0)

    def test_fresh_symbol_failing_gates_does_not_raise(self):
        """新しい足がゲートで落ちただけなら、取得は止まっておらず該当なしとする。"""
        df_map = {
            ("us", "STOP"): _uptrend(last_day=date(2026, 10, 1)),
            ("us", "DOWN"): _downtrend(),
        }
        self.assertEqual(self._codes(df_map), [])


class TestComputeMetricsInvalidCloses(unittest.TestCase):
    """終値が NaN・無限・0以下の足を除いて計算する（#1597）。"""

    def _assert_same_as_without_bar(self, position: int, bad_value: float) -> None:
        closes = np.linspace(10.0, 100.0, _N)
        closes[position] = bad_value
        df = _make_df(closes)
        self.assertEqual(_compute_metrics(df), _compute_metrics(df.drop(index=df.index[position])))

    def test_last_nan_close_uses_last_valid_close(self):
        """最後の足が NaN なら、残った足の最後の終値で計算する。"""
        closes = np.linspace(10.0, 100.0, _N)
        last_valid = float(closes[-2])
        closes[-1] = np.nan
        metrics = _compute_metrics(_make_df(closes))
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["close"], last_valid)

    def test_zero_close_at_6m_position_is_dropped(self):
        """-126 の位置の終値が 0 でも例外を出さず、その足を除いて計算する。"""
        self._assert_same_as_without_bar(-126, 0.0)

    def test_zero_close_at_12m_position_is_dropped(self):
        """-252 の位置の終値が 0 でも例外を出さず、その足を除いて計算する。"""
        self._assert_same_as_without_bar(-252, 0.0)

    def test_negative_close_is_dropped(self):
        """負の終値の足を除いて計算する。"""
        self._assert_same_as_without_bar(-50, -1.0)

    def test_infinite_close_is_dropped(self):
        """無限の終値の足を除いて計算する（高値が無限にならない）。"""
        self._assert_same_as_without_bar(-50, np.inf)

    def test_too_few_valid_closes_returns_none(self):
        """有効な足が252本に満たなければ None を返す。"""
        closes = np.linspace(10.0, 100.0, 252)
        closes[100] = np.nan
        self.assertIsNone(_compute_metrics(_make_df(closes)))

    def test_exactly_252_valid_closes_is_computed(self):
        """無効な足を除いて252本ちょうど残れば計算する。"""
        closes = np.linspace(10.0, 100.0, 253)
        closes[100] = np.nan
        self.assertIsNotNone(_compute_metrics(_make_df(closes)))

    def test_clean_series_metrics_match_direct_calculation(self):
        """欠けの無い系列の指標は、全足からじかに計算した値と同じである。"""
        closes = np.linspace(10.0, 100.0, _N)
        last = float(closes[-1])
        sma200 = float(closes[-200:].mean())
        high52 = float(closes[-252:].max())
        expected = {
            "close": last,
            "dist_from_52w_high": last / high52 - 1.0,
            "above_200dma": True,
            "sma200_rising": True,
            "return_6m": last / float(closes[-126]) - 1.0,
            "return_12m": last / float(closes[-252]) - 1.0,
            "avg_volume": 1_000_000.0,
            "trend_strength": last / sma200 - 1.0,
        }
        metrics = _compute_metrics(_make_df(closes))
        self.assertIsNotNone(metrics)
        self.assertEqual(set(metrics), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
