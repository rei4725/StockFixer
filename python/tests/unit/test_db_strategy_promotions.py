import os
import tempfile
import unittest
from datetime import datetime

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.utils.db.strategy_promotions import (
    load_active_promotions,
    mark_promotion_rolled_back,
    promotion_exists,
    save_strategy_promotion,
)


class _TmpDbTestCase(unittest.TestCase):
    def setUp(self):
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path


class TestStrategyPromotionsDb(_TmpDbTestCase):
    def test_save_and_load_roundtrip(self):
        save_strategy_promotion(
            pr_number=101,
            merge_commit_hash="abc123",
            rule_or_feature_id="fb44f0011174",
            pre_promotion_baseline=1.25,
        )
        df = load_active_promotions()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["pr_number"], 101)
        self.assertEqual(df.iloc[0]["merge_commit_hash"], "abc123")
        self.assertEqual(df.iloc[0]["rule_or_feature_id"], "fb44f0011174")
        self.assertAlmostEqual(float(df.iloc[0]["pre_promotion_baseline"]), 1.25)
        self.assertEqual(df.iloc[0]["status"], "active")

    def test_promotion_exists(self):
        self.assertFalse(promotion_exists(202))
        save_strategy_promotion(
            pr_number=202,
            merge_commit_hash="def456",
            rule_or_feature_id="hash2",
            pre_promotion_baseline=0.9,
        )
        self.assertTrue(promotion_exists(202))

    def test_mark_rolled_back_excludes_from_active(self):
        save_strategy_promotion(
            pr_number=303,
            merge_commit_hash="ghi789",
            rule_or_feature_id="hash3",
            pre_promotion_baseline=1.1,
        )
        mark_promotion_rolled_back(303)
        df = load_active_promotions()
        self.assertEqual(len(df), 0)

    def test_promoted_at_defaults_to_now(self):
        before = datetime.now()
        save_strategy_promotion(
            pr_number=404,
            merge_commit_hash="jkl012",
            rule_or_feature_id="hash4",
            pre_promotion_baseline=1.0,
        )
        df = load_active_promotions()
        promoted_at = df.iloc[0]["promoted_at"]
        self.assertGreaterEqual(promoted_at, before)

    def test_duplicate_pr_number_is_replaced_not_duplicated(self):
        save_strategy_promotion(
            pr_number=505,
            merge_commit_hash="first",
            rule_or_feature_id="hash5",
            pre_promotion_baseline=1.0,
        )
        save_strategy_promotion(
            pr_number=505,
            merge_commit_hash="second",
            rule_or_feature_id="hash5",
            pre_promotion_baseline=1.0,
        )
        df = load_active_promotions()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["merge_commit_hash"], "second")
