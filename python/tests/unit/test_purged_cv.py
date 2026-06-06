"""Purged K-Fold + Embargo CV の単体テスト（#372）。

リーク排除の不変条件（train/test 非重複・purge・embargo）を検証する。
"""

import numpy as np
import pytest

from src.prediction.purged_cv import PurgedKFold, purged_kfold_indices


class TestPurgedKFold:
    def test_invalid_n_splits(self):
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)

    def test_get_n_splits(self):
        assert PurgedKFold(n_splits=4).get_n_splits() == 4

    def test_folds_cover_and_are_disjoint(self):
        n = 100
        splits = purged_kfold_indices(n, n_splits=5, embargo_pct=0.0, purge_gap=0)
        assert len(splits) == 5
        test_all = np.concatenate([te for _, te in splits])
        # テスト集合は全体を重複なく被覆する
        assert sorted(test_all.tolist()) == list(range(n))

    def test_train_test_no_overlap(self):
        n = 200
        for tr, te in purged_kfold_indices(n, n_splits=5, embargo_pct=0.02, purge_gap=2):
            assert set(tr.tolist()).isdisjoint(set(te.tolist()))

    def test_purge_removes_pre_test_samples(self):
        """テスト開始直前 purge_gap 本が訓練から除外される。"""
        n = 100
        gap = 3
        splits = purged_kfold_indices(n, n_splits=5, embargo_pct=0.0, purge_gap=gap)
        # 2 番目のフォールド（先頭以外）を検証
        tr, te = splits[1]
        t0 = int(te.min())
        purged = set(range(max(0, t0 - gap), t0))
        assert purged.isdisjoint(set(tr.tolist()))

    def test_embargo_removes_post_test_samples(self):
        """テスト直後 embargo 本が訓練から除外される。"""
        n = 100
        emb_pct = 0.05  # embargo = 5
        splits = purged_kfold_indices(n, n_splits=5, embargo_pct=emb_pct, purge_gap=0)
        tr, te = splits[1]
        t1 = int(te.max()) + 1
        embargoed = set(range(t1, min(n, t1 + int(n * emb_pct))))
        assert embargoed.isdisjoint(set(tr.tolist()))

    def test_no_purge_no_embargo_is_plain_kfold(self):
        """purge=0, embargo=0 なら train は test の補集合（連続 KFold）。"""
        n = 50
        for tr, te in purged_kfold_indices(n, n_splits=5, embargo_pct=0.0, purge_gap=0):
            assert sorted(tr.tolist() + te.tolist()) == list(range(n))

    def test_raises_when_too_few_samples(self):
        with pytest.raises(ValueError):
            list(PurgedKFold(n_splits=5).split(np.empty((3, 1))))
