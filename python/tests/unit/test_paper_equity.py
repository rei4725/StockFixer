"""paper_equity / equity_chart のユニットテスト（月次損益曲線レポート）"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.reporting.equity_chart import _is_benchmark, build_equity_chart
from src.trading.paper_equity import (
    _default_split_loader,
    build_equity_series,
    get_paper_equity_curve,
)

_INITIAL = 1_000_000.0


def _index(n=10, start="2026-01-05"):
    return pd.bdate_range(start=start, periods=n)


def _orders(rows):
    return pd.DataFrame(
        rows, columns=["market", "symbol", "side", "qty", "fill_price", "filled_at"]
    )


def _flat_price_loader(price=100.0):
    def loader(market, symbol):
        idx = pd.bdate_range("2025-12-01", periods=60)
        return pd.Series(price, index=idx)

    return loader


class TestBuildEquitySeries(unittest.TestCase):
    """build_equity_series（純粋ロジック）のテスト"""

    def test_empty_index_returns_empty(self):
        result = build_equity_series(
            _orders([]), _flat_price_loader(), _INITIAL, pd.DatetimeIndex([])
        )
        self.assertTrue(result.empty)

    def test_buy_and_hold_marks_to_market(self):
        """買い建て後は equity = 現金 + 保有数量 × 終値 になること"""
        idx = _index()
        orders = _orders([("jp", "7203", 1, 100, 100.0, idx[1])])

        equity = build_equity_series(orders, _flat_price_loader(100.0), _INITIAL, idx)

        # 約定日前は初期資金のまま
        self.assertAlmostEqual(equity.iloc[0], _INITIAL)
        # 約定日以降: 現金 -10,000 + 評価 100株×100円 = 差し引きゼロ（手数料モデルなし）
        self.assertAlmostEqual(equity.iloc[1], _INITIAL)
        self.assertAlmostEqual(equity.iloc[-1], _INITIAL)

    def test_price_appreciation_increases_equity(self):
        """保有中の値上がりが評価額に反映されること"""
        idx = _index()
        orders = _orders([("jp", "7203", 1, 100, 100.0, idx[0])])

        def rising_loader(market, symbol):
            full = pd.bdate_range("2025-12-01", periods=60)
            prices = pd.Series(100.0, index=full)
            prices[prices.index >= idx[5]] = 120.0
            return prices

        equity = build_equity_series(orders, rising_loader, _INITIAL, idx)

        self.assertAlmostEqual(equity.iloc[0], _INITIAL)
        # 値上がり後: +100株 × 20円 = +2,000
        self.assertAlmostEqual(equity.iloc[-1], _INITIAL + 2_000.0)

    def test_round_trip_realizes_pnl_in_cash(self):
        """買い→売りの往復後は保有ゼロで損益が現金に反映されること"""
        idx = _index()
        orders = _orders(
            [
                ("jp", "7203", 1, 100, 100.0, idx[0]),
                ("jp", "7203", 2, 100, 110.0, idx[3]),
            ]
        )

        equity = build_equity_series(orders, _flat_price_loader(110.0), _INITIAL, idx)

        # 売却後: 初期資金 + (110-100)×100 = +1,000、以後フラット
        self.assertAlmostEqual(equity.iloc[-1], _INITIAL + 1_000.0)
        self.assertAlmostEqual(equity.iloc[3], equity.iloc[-1])

    def test_fill_before_index_is_applied_at_first_day(self):
        """index 開始前の約定は初日に寄せて反映されること"""
        idx = _index(start="2026-02-02")
        orders = _orders([("jp", "7203", 1, 50, 200.0, pd.Timestamp("2026-01-15"))])

        equity = build_equity_series(orders, _flat_price_loader(200.0), _INITIAL, idx)

        self.assertAlmostEqual(equity.iloc[0], _INITIAL)  # 現金 -10,000 + 評価 +10,000

    def test_missing_price_values_at_fill_price(self):
        """価格データのない銘柄は0円でなく約定の値段で評価されること（#850）"""
        idx = _index()
        orders = _orders([("jp", "9999", 1, 10, 500.0, idx[0])])

        equity = build_equity_series(orders, lambda m, s: None, _INITIAL, idx)

        # 現金 -5,000 + 10株 × 約定 500円 = 差し引きゼロ
        self.assertAlmostEqual(equity.iloc[0], _INITIAL)
        self.assertAlmostEqual(equity.iloc[-1], _INITIAL)

    def test_price_series_starting_after_buy_does_not_use_later_price(self):
        """値段の系列が買った日より後に始まるとき、その前の保有日を後の値段で埋めないこと"""
        idx = _index()
        orders = _orders([("jp", "7203", 1, 10, 500.0, idx[0])])

        def late_loader(market, symbol):
            return pd.Series(600.0, index=idx[3:])

        equity = build_equity_series(orders, late_loader, _INITIAL, idx)

        # 値段の無い保有日は約定の 500円、値段が出てからは 600円
        self.assertAlmostEqual(equity.iloc[0], _INITIAL)
        self.assertAlmostEqual(equity.iloc[2], _INITIAL)
        self.assertAlmostEqual(equity.iloc[3], _INITIAL + 1_000.0)
        self.assertAlmostEqual(equity.iloc[-1], _INITIAL + 1_000.0)

    def test_price_gap_uses_last_known_price(self):
        """値段の系列が途中で切れたら、最後に分かる値段で評価すること"""
        idx = _index()
        orders = _orders([("jp", "7203", 1, 100, 100.0, idx[0])])

        def ending_loader(market, symbol):
            prices = pd.Series(100.0, index=idx[:5])
            prices[idx[4]] = 120.0
            return prices

        equity = build_equity_series(orders, ending_loader, _INITIAL, idx)

        self.assertAlmostEqual(equity.iloc[-1], _INITIAL + 2_000.0)

    def test_short_affects_cash_only(self):
        """SHORT（side=3）は現金にのみ反映され保有評価を持たないこと"""
        idx = _index()
        orders = _orders([("jp", "7203", 3, 100, 100.0, idx[0])])

        equity = build_equity_series(orders, _flat_price_loader(100.0), _INITIAL, idx)

        self.assertAlmostEqual(equity.iloc[-1], _INITIAL + 10_000.0)


class TestBuildEquitySeriesSplit(unittest.TestCase):
    """保有の途中に分割があり、raw の終値が全期間で調整し直してあるときの評価（#899）"""

    def setUp(self):
        self.idx = _index()
        self.split_day = self.idx[5]

    def _splits(self, market, symbol):
        return pd.Series([2.0], index=[self.split_day])

    def test_split_adjusted_close_does_not_halve_holding(self):
        """分割前に 100 円で買った 100 株は、調整後の 50 円で 200 株として評価すること"""
        orders = _orders([("jp", "7203", 1, 100, 100.0, self.idx[0])])

        equity = build_equity_series(
            orders, _flat_price_loader(50.0), _INITIAL, self.idx, split_loader=self._splits
        )

        # 実際の値動きは無いので、分割の前後とも元手のまま（段差が無い）
        self.assertAlmostEqual(equity.iloc[0], _INITIAL)
        self.assertAlmostEqual(equity.iloc[4], _INITIAL)
        self.assertAlmostEqual(equity.iloc[5], _INITIAL)
        self.assertAlmostEqual(equity.iloc[-1], _INITIAL)

    def test_sell_after_split_closes_position(self):
        """分割後に 200 株を売ると、分割前の 100 株の保有が消えること"""
        orders = _orders(
            [
                ("jp", "7203", 1, 100, 100.0, self.idx[0]),
                ("jp", "7203", 2, 200, 55.0, self.idx[7]),
            ]
        )

        equity = build_equity_series(
            orders, _flat_price_loader(50.0), _INITIAL, self.idx, split_loader=self._splits
        )

        # 売った後: 現金 -10,000 + 11,000 = +1,000、保有は残らない
        self.assertAlmostEqual(equity.iloc[7], _INITIAL + 1_000.0)
        self.assertAlmostEqual(equity.iloc[-1], _INITIAL + 1_000.0)

    def test_fill_on_split_day_is_not_adjusted(self):
        """分割の日の約定は、もう分割後の水準なので直さないこと"""
        orders = _orders([("jp", "7203", 1, 100, 50.0, self.split_day)])

        equity = build_equity_series(
            orders, _flat_price_loader(60.0), _INITIAL, self.idx, split_loader=self._splits
        )

        self.assertAlmostEqual(equity.iloc[-1], _INITIAL + 1_000.0)

    def test_missing_price_falls_back_to_adjusted_fill_price(self):
        """終値の無い日は、分割の比で割った約定の値段で評価すること"""
        orders = _orders([("jp", "7203", 1, 100, 100.0, self.idx[0])])

        equity = build_equity_series(
            orders, lambda m, s: None, _INITIAL, self.idx, split_loader=self._splits
        )

        self.assertAlmostEqual(equity.iloc[-1], _INITIAL)

    def test_no_split_events_keeps_valuation(self):
        """分割の記録が無い銘柄の評価は変わらないこと"""
        orders = _orders([("jp", "7203", 1, 100, 100.0, self.idx[0])])

        equity = build_equity_series(
            orders, _flat_price_loader(120.0), _INITIAL, self.idx, split_loader=lambda m, s: None
        )

        self.assertAlmostEqual(equity.iloc[-1], _INITIAL + 2_000.0)


class TestDefaultSplitLoader(unittest.TestCase):
    """_default_split_loader（yfinance の分割の記録）のテスト"""

    @patch("src.trading.paper_equity.yf.Ticker")
    def test_returns_naive_normalized_index(self, mock_ticker):
        idx = pd.DatetimeIndex(["2026-01-12 00:00"]).tz_localize("Asia/Tokyo")
        mock_ticker.return_value.splits = pd.Series([2.0], index=idx)

        result = _default_split_loader("jp", "7203")

        self.assertEqual(list(result.index), [pd.Timestamp("2026-01-12")])
        self.assertEqual(list(result), [2.0])

    @patch("src.trading.paper_equity.yf.Ticker")
    def test_empty_splits_returns_none(self, mock_ticker):
        mock_ticker.return_value.splits = pd.Series(dtype=float)

        self.assertIsNone(_default_split_loader("us", "AAPL"))

    @patch("src.trading.paper_equity.yf.Ticker", side_effect=RuntimeError("network"))
    def test_error_returns_none(self, _mock_ticker):
        self.assertIsNone(_default_split_loader("us", "AAPL"))


class TestGetPaperEquityCurve(unittest.TestCase):
    """get_paper_equity_curve のテスト"""

    @patch("src.trading.paper_equity._load_filled_orders")
    def test_empty_orders_returns_empty_series(self, mock_load):
        mock_load.return_value = pd.DataFrame(
            columns=["market", "symbol", "side", "qty", "fill_price", "filled_at"]
        )
        result = get_paper_equity_curve()
        self.assertTrue(result.empty)


class TestBuildEquityChart(unittest.TestCase):
    """build_equity_chart のテスト"""

    def _equity(self, n=30):
        idx = pd.bdate_range("2026-01-05", periods=n)
        rng = np.random.default_rng(42)
        return pd.Series(_INITIAL * np.exp(np.cumsum(rng.normal(0.001, 0.01, n))), index=idx)

    def test_writes_png_with_benchmark(self):
        equity = self._equity()
        benchmark = pd.Series(np.linspace(5000, 5200, len(equity)), index=equity.index)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "equity.png")
            path = build_equity_chart({"Paper Trading": equity, "S&P 500": benchmark}, out)
            self.assertEqual(path, out)
            self.assertGreater(os.path.getsize(out), 1000)

    def test_writes_png_without_benchmark(self):
        equity = self._equity()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "equity.png")
            build_equity_chart({"Paper Trading": equity}, out)
            self.assertTrue(os.path.exists(out))

    def test_writes_png_with_three_series(self):
        equity = self._equity()
        allocation = self._equity() * 0.5
        benchmark = pd.Series(np.linspace(5000, 5200, len(equity)), index=equity.index)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "equity.png")
            build_equity_chart(
                {"Paper Trading": equity, "Allocation Bot": allocation, "S&P 500": benchmark},
                out,
            )
            self.assertGreater(os.path.getsize(out), 1000)

    def test_skips_empty_series_but_still_renders(self):
        equity = self._equity()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "equity.png")
            build_equity_chart(
                {"Paper Trading": equity, "Allocation Bot": pd.Series(dtype=float)}, out
            )
            self.assertTrue(os.path.exists(out))

    def test_raises_on_empty_dict(self):
        with self.assertRaises(ValueError):
            build_equity_chart({}, "unused.png")

    def test_raises_when_all_series_empty(self):
        with self.assertRaises(ValueError):
            build_equity_chart({"Paper Trading": pd.Series(dtype=float)}, "unused.png")


class TestIsBenchmark(unittest.TestCase):
    """_is_benchmark（開始日サフィックス付きラベルの前方一致判定）のテスト"""

    def test_exact_label_matches(self):
        self.assertTrue(_is_benchmark("S&P 500", frozenset({"S&P 500"})))

    def test_label_with_start_date_suffix_matches(self):
        self.assertTrue(_is_benchmark("S&P 500 (since 2026-01-05)", frozenset({"S&P 500"})))

    def test_unrelated_label_does_not_match(self):
        self.assertFalse(_is_benchmark("Allocation Bot (since 2026-01-05)", frozenset({"S&P 500"})))

    def test_accidental_prefix_does_not_match(self):
        self.assertFalse(_is_benchmark("S&P 500X", frozenset({"S&P 500"})))


if __name__ == "__main__":
    unittest.main()
