"""過学習ガード（PBO / CSCV）の単体テスト（#372）。

PBO は「インサンプル最良戦略が OOS で中央値を下回る確率」。
- 全戦略がノイズ（同分布）＝最適化が過学習 → PBO は高い（~0.5）
- 1戦略だけ持続的エッジ＝頑健 → PBO は低い（~0）
"""

import math

import numpy as np

from src.backtest.metrics import _sharpe_by_column, probability_of_backtest_overfitting


class TestSharpeByColumn:
    def test_per_column_sharpe(self):
        block = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
        # 列0: mean=1 std=0 → 0、列1も std=0 → 0（std=0 は 0 扱い）
        out = _sharpe_by_column(block)
        assert out.shape == (2,)
        assert np.all(out == 0.0)

    def test_positive_mean_positive_sharpe(self):
        rng = np.random.default_rng(1)
        block = rng.normal(0.5, 1.0, size=(200, 1))
        assert _sharpe_by_column(block)[0] > 0


class TestPBO:
    def test_noise_gives_high_pbo(self):
        """全戦略が同分布のノイズ → 過学習 → PBO は 0.5 付近（>=0.4）。"""
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.01, size=(1000, 20))
        pbo = probability_of_backtest_overfitting(noise, n_splits=10)
        assert 0.4 <= pbo <= 0.6

    def test_true_edge_gives_low_pbo(self):
        """1戦略だけ持続的エッジ → IS最良が OOS でも上位 → PBO は低い（<0.2）。"""
        rng = np.random.default_rng(0)
        m = rng.normal(0.0, 0.01, size=(1000, 20))
        m[:, 0] += 0.004  # 戦略0に持続的エッジ
        pbo = probability_of_backtest_overfitting(m, n_splits=10)
        assert pbo < 0.2

    def test_single_candidate_returns_nan(self):
        rng = np.random.default_rng(0)
        out = probability_of_backtest_overfitting(rng.normal(size=(500, 1)))
        assert math.isnan(out)

    def test_insufficient_data_returns_nan(self):
        # 期間がブロック数より少ない
        out = probability_of_backtest_overfitting(np.zeros((4, 5)), n_splits=10)
        assert math.isnan(out)

    def test_odd_n_splits_is_handled(self):
        rng = np.random.default_rng(2)
        noise = rng.normal(0.0, 0.01, size=(600, 10))
        # 奇数指定でも内部で偶数に丸めて計算できる
        pbo = probability_of_backtest_overfitting(noise, n_splits=9)
        assert 0.0 <= pbo <= 1.0

    def test_pbo_in_unit_range(self):
        rng = np.random.default_rng(3)
        m = rng.normal(0.0, 0.01, size=(800, 8))
        pbo = probability_of_backtest_overfitting(m, n_splits=8)
        assert 0.0 <= pbo <= 1.0
