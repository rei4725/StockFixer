"""時系列リークを排除した Purged K-Fold + Embargo クロスバリデーション（#372）。

通常の KFold は時系列データでリーク（未来情報の混入）を起こす。López de Prado の
Purged K-Fold は、時間順サンプルを K 個の連続フォールドに分割し、各フォールドを
テストにする際、訓練集合から次を除外する:

- **Purge**: テスト開始直前の ``purge_gap`` 本。前向きラベル（t の目的変数が [t, t+h]
  を使う）がテスト期間と重なる訓練サンプルを取り除く。
- **Embargo**: テスト直後の ``embargo`` 本。テスト末尾のラベルが直後の訓練サンプルと
  相関してリークするのを防ぐ。

sklearn 互換の ``split`` / ``get_n_splits`` を提供し、cross_val 系にそのまま渡せる。
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np


class PurgedKFold:
    """Purged + Embargo K-Fold スプリッタ（時系列リーク排除）。

    Args:
        n_splits: フォールド数（>=2）。
        embargo_pct: テスト直後に訓練から除外する割合（全長 n に対する比率）。
        purge_gap: テスト開始直前に訓練から除外する本数（ラベル地平 h に相当）。
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01, purge_gap: int = 1):
        if n_splits < 2:
            raise ValueError("n_splits は 2 以上が必要です")
        if embargo_pct < 0:
            raise ValueError("embargo_pct は 0 以上が必要です")
        if purge_gap < 0:
            raise ValueError("purge_gap は 0 以上が必要です")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_gap = purge_gap

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """時間順に並んだ X を分割し (train_idx, test_idx) を順に返す。

        X は時系列順（古い→新しい）に並んでいることを前提とする。
        """
        n = int(X.shape[0]) if hasattr(X, "shape") else len(X)
        if n < self.n_splits:
            raise ValueError("サンプル数が n_splits より少ないため分割できません")

        indices = np.arange(n)
        embargo = int(n * self.embargo_pct)
        bounds = np.linspace(0, n, self.n_splits + 1).astype(int)

        for k in range(self.n_splits):
            t0, t1 = int(bounds[k]), int(bounds[k + 1])
            if t1 <= t0:
                continue
            test_idx = indices[t0:t1]

            train_mask = np.ones(n, dtype=bool)
            # テスト区間自体を除外
            train_mask[t0:t1] = False
            # Purge: テスト開始直前 purge_gap 本
            lo = max(0, t0 - self.purge_gap)
            train_mask[lo:t0] = False
            # Embargo: テスト直後 embargo 本
            hi = min(n, t1 + embargo)
            train_mask[t1:hi] = False

            train_idx = indices[train_mask]
            if len(train_idx) == 0:
                continue
            yield train_idx, test_idx


def purged_kfold_indices(
    n: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    purge_gap: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """長さ n の時系列に対する (train_idx, test_idx) のリストを返す簡易ヘルパ。"""
    splitter = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct, purge_gap=purge_gap)
    dummy: Optional[np.ndarray] = np.empty((n, 1))
    return list(splitter.split(dummy))
