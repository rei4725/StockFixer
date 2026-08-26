"""ユニットテスト: ATRベースの予測変化率クリップ（range_clipping.py）"""

from __future__ import annotations

import math
import unittest

from src.prediction.range_clipping import clip_diff_ratio_to_atr_range


class TestClipDiffRatioToAtrRange(unittest.TestCase):
    def test_within_range_is_unchanged(self):
        # max_ratio = 3.0 * (10/1000) = 0.03、diff_ratio=0.01 は範囲内
        result = clip_diff_ratio_to_atr_range(0.01, atr=10.0, current_price=1000.0)
        self.assertAlmostEqual(result, 0.01)

    def test_positive_outlier_is_clipped_to_max(self):
        # max_ratio = 3.0 * (10/1000) = 0.03、diff_ratio=0.50 は外れ値
        result = clip_diff_ratio_to_atr_range(0.50, atr=10.0, current_price=1000.0)
        self.assertAlmostEqual(result, 0.03)

    def test_negative_outlier_is_clipped_to_min(self):
        result = clip_diff_ratio_to_atr_range(-0.50, atr=10.0, current_price=1000.0)
        self.assertAlmostEqual(result, -0.03)

    def test_atr_none_returns_unchanged(self):
        result = clip_diff_ratio_to_atr_range(0.50, atr=None, current_price=1000.0)
        self.assertAlmostEqual(result, 0.50)

    def test_atr_zero_returns_unchanged(self):
        result = clip_diff_ratio_to_atr_range(0.50, atr=0.0, current_price=1000.0)
        self.assertAlmostEqual(result, 0.50)

    def test_atr_negative_returns_unchanged(self):
        result = clip_diff_ratio_to_atr_range(0.50, atr=-5.0, current_price=1000.0)
        self.assertAlmostEqual(result, 0.50)

    def test_current_price_zero_returns_unchanged(self):
        result = clip_diff_ratio_to_atr_range(0.50, atr=10.0, current_price=0.0)
        self.assertAlmostEqual(result, 0.50)

    def test_current_price_negative_returns_unchanged(self):
        result = clip_diff_ratio_to_atr_range(0.50, atr=10.0, current_price=-100.0)
        self.assertAlmostEqual(result, 0.50)

    def test_custom_atr_multiplier_widens_range(self):
        result = clip_diff_ratio_to_atr_range(
            0.50, atr=10.0, current_price=1000.0, atr_multiplier=10.0
        )
        # max_ratio = 10.0 * (10/1000) = 0.10 なのでまだクリップされる
        self.assertAlmostEqual(result, 0.10)

    def test_horizon_scales_range_by_sqrt(self):
        # horizon=1: max_ratio=0.03, horizon=4: max_ratio=0.03*sqrt(4)=0.06
        result_h1 = clip_diff_ratio_to_atr_range(0.50, atr=10.0, current_price=1000.0, horizon=1)
        result_h4 = clip_diff_ratio_to_atr_range(0.50, atr=10.0, current_price=1000.0, horizon=4)
        self.assertAlmostEqual(result_h1, 0.03)
        self.assertAlmostEqual(result_h4, 0.03 * math.sqrt(4))
        self.assertGreater(result_h4, result_h1)

    def test_boundary_value_is_not_clipped(self):
        # diff_ratio がちょうど max_ratio の場合はクリップされない（境界値）
        max_ratio = 3.0 * (10.0 / 1000.0)
        result = clip_diff_ratio_to_atr_range(max_ratio, atr=10.0, current_price=1000.0)
        self.assertAlmostEqual(result, max_ratio)

    def test_zero_diff_ratio_unchanged(self):
        result = clip_diff_ratio_to_atr_range(0.0, atr=10.0, current_price=1000.0)
        self.assertAlmostEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
