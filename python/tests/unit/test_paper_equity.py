"""paper_equity / equity_chart のユニットテスト（月次損益曲線レポート）"""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.reporting.equity_chart import build_equity_chart
from src.trading.paper_equity import build_equity_series, get_paper_equity_curve

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

    def test_missing_price_excludes_position(self):
        """価格データのない銘柄は評価から除外され現金のみになること"""
        idx = _index()
        orders = _orders([("jp", "9999", 1, 10, 500.0, idx[0])])

        equity = build_equity_series(orders, lambda m, s: None, _INITIAL, idx)

        self.assertAlmostEqual(equity.iloc[-1], _INITIAL - 5_000.0)

    def test_short_affects_cash_only(self):
        """SHORT（side=3）は現金にのみ反映され保有評価を持たないこと"""
        idx = _index()
        orders = _orders([("jp", "7203", 3, 100, 100.0, idx[0])])

        equity = build_equity_series(orders, _flat_price_loader(100.0), _INITIAL, idx)

        self.assertAlmostEqual(equity.iloc[-1], _INITIAL + 10_000.0)


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
            path = build_equity_chart(equity, benchmark, out)
            self.assertEqual(path, out)
            self.assertGreater(os.path.getsize(out), 1000)

    def test_writes_png_without_benchmark(self):
        equity = self._equity()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "equity.png")
            build_equity_chart(equity, None, out)
            self.assertTrue(os.path.exists(out))

    def test_raises_on_empty_equity(self):
        with self.assertRaises(ValueError):
            build_equity_chart(pd.Series(dtype=float), None, "unused.png")


if __name__ == "__main__":
    unittest.main()
