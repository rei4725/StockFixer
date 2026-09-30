"""ポートフォリオ equity 合成とドローダウンのテスト（#703 後続 / ゲート指標の是正）。

最悪銘柄DD（min over symbols）ではなく、有効銘柄を等金額で保有した場合の
ポートフォリオDDをゲート指標にするための合成ロジックを検証する。
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.backtest.factory_portfolio import build_portfolio_equity, portfolio_max_drawdown

_INITIAL = 100.0


def _curve(dates: list[str], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates), dtype=float)


def _reference_build_portfolio_equity(curves: list[pd.Series], initial_cash: float) -> pd.Series:
    """numpy 化する前の実装（pandas の union / reindex）。等価性の検証用に残す。"""
    usable = [c for c in curves if c is not None and not c.empty]
    if not usable or initial_cash <= 0:
        return pd.Series(dtype=float)
    normalized = [c.astype(float) / initial_cash for c in usable]
    index = normalized[0].index
    for curve in normalized[1:]:
        index = index.union(curve.index)
    index = index.sort_values()
    aligned = [c.reindex(index).ffill().fillna(1.0) for c in normalized]
    return pd.concat(aligned, axis=1).mean(axis=1)


class TestBuildPortfolioEquityMatchesReference(unittest.TestCase):
    """numpy 実装が従来の pandas 実装と完全に一致すること（開始日・終了日・欠損日がばらばら）。"""

    def test_matches_reference_on_ragged_random_curves(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            curves = []
            for _ in range(int(rng.integers(1, 30))):
                start = int(rng.integers(0, 150))
                end = int(rng.integers(start + 1, 300))
                idx = dates[start:end]
                keep = rng.random(len(idx)) > 0.1  # 記録の隙間
                idx = idx[keep] if keep.any() else idx[:1]
                values = _INITIAL * np.cumprod(1 + rng.normal(0, 0.02, len(idx)))
                curves.append(pd.Series(values, index=idx))
            curves.append(pd.Series(dtype=float))  # 空曲線は無視される
            expected = _reference_build_portfolio_equity(curves, _INITIAL)
            actual = build_portfolio_equity(curves, _INITIAL)
            pd.testing.assert_series_equal(actual, expected, check_names=False, check_freq=False)

    def test_matches_reference_when_all_curves_share_dates(self):
        """全銘柄が同じ日付列の近道（突き合わせ省略）でも一致すること。NaN を含めば通常経路。"""
        dates = pd.to_datetime(pd.bdate_range("2024-01-01", periods=120).to_numpy())
        rng = np.random.default_rng(0)
        curves = [
            pd.Series(_INITIAL * np.cumprod(1 + rng.normal(0, 0.02, 120)), index=dates)
            for _ in range(10)
        ]
        with_gap = [c.copy() for c in curves]
        with_gap[3].iloc[40:45] = np.nan
        for case in (curves, with_gap):
            expected = _reference_build_portfolio_equity(case, _INITIAL)
            actual = build_portfolio_equity(case, _INITIAL)
            pd.testing.assert_series_equal(actual, expected, check_names=False, check_freq=False)


class TestBuildPortfolioEquity(unittest.TestCase):
    def test_returns_empty_series_when_no_curves(self):
        result = build_portfolio_equity([], _INITIAL)

        self.assertTrue(result.empty)

    def test_single_curve_is_normalized_by_initial_cash(self):
        curve = _curve(["2024-01-01", "2024-01-02"], [100.0, 120.0])

        result = build_portfolio_equity([curve], _INITIAL)

        self.assertEqual(list(result.values), [1.0, 1.2])

    def test_equal_weight_average_of_two_aligned_curves(self):
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        a = _curve(dates, [100.0, 50.0, 100.0, 100.0])
        b = _curve(dates, [100.0, 150.0, 150.0, 150.0])

        result = build_portfolio_equity([a, b], _INITIAL)

        self.assertEqual(list(result.values), [1.0, 1.0, 1.25, 1.25])

    def test_unaligned_dates_are_forward_filled_and_prefilled_with_cash(self):
        # AAA は d1〜d3、BBB は d2〜d4 にしか記録がない。
        # BBB の d1 は「まだ建玉がなく現金のまま」= 1.0、AAA の d4 は前方補完で 0.8。
        a = _curve(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 80.0, 80.0])
        b = _curve(["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 100.0, 120.0])

        result = build_portfolio_equity([a, b], _INITIAL)

        self.assertEqual(
            list(result.index),
            list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])),
        )
        self.assertEqual(list(result.values), [1.0, 0.9, 0.9, 1.0])

    def test_empty_curves_are_ignored(self):
        a = _curve(["2024-01-01", "2024-01-02"], [100.0, 120.0])
        empty = pd.Series(dtype=float)

        result = build_portfolio_equity([a, empty], _INITIAL)

        self.assertEqual(list(result.values), [1.0, 1.2])


class TestPortfolioMaxDrawdown(unittest.TestCase):
    def test_is_shallower_than_worst_symbol_drawdown(self):
        dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
        # AAA 単体の DD は -50%、BBB は 0%。等金額ポートフォリオでは相殺されて 0%。
        a = _curve(dates, [100.0, 50.0, 100.0, 100.0])
        b = _curve(dates, [100.0, 150.0, 150.0, 150.0])

        result = portfolio_max_drawdown([a, b], _INITIAL)

        self.assertAlmostEqual(result, 0.0)

    def test_drawdown_of_partially_overlapping_curves(self):
        a = _curve(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 80.0, 80.0])
        b = _curve(["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 100.0, 120.0])

        result = portfolio_max_drawdown([a, b], _INITIAL)

        self.assertAlmostEqual(result, -0.1)

    def test_returns_nan_when_no_usable_curves(self):
        # 呼び出し側が「最悪銘柄DDへフォールバックすべき」と判別できるよう NaN を返す。
        self.assertTrue(math.isnan(portfolio_max_drawdown([], _INITIAL)))
        self.assertTrue(math.isnan(portfolio_max_drawdown([pd.Series(dtype=float)], _INITIAL)))


if __name__ == "__main__":
    unittest.main()
