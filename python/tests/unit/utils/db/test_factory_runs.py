import os
import tempfile
import unittest

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.utils.db._connection import _db_connection
from src.utils.db.factory_runs import count_factory_runs, load_factory_hashes, save_factory_run


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


class TestFactoryRunsDb(_TmpDbTestCase):
    def _make_run(self, hypothesis_hash="hash-1", **overrides):
        kwargs = dict(
            hypothesis_hash=hypothesis_hash,
            market="us",
            spec_json='{"rule": "example"}',
            sharpe_ratio=1.2,
            win_rate=0.55,
            num_trades=42,
            max_drawdown=-0.1,
            total_return=0.3,
            dsr=0.8,
            pbo=0.2,
            gate_passed=True,
            gate_reasons=None,
            report_path=None,
        )
        kwargs.update(overrides)
        save_factory_run(**kwargs)

    def test_save_and_load_roundtrip(self):
        self._make_run(hypothesis_hash="round-trip-hash")
        hashes = load_factory_hashes()
        self.assertIn("round-trip-hash", hashes)

        with _db_connection() as con:
            row = con.execute(
                """
                SELECT hypothesis_hash, market, spec_json, sharpe_ratio, win_rate,
                       num_trades, max_drawdown, total_return, dsr, pbo, gate_passed
                FROM factory_runs WHERE hypothesis_hash = %s
                """,
                ["round-trip-hash"],
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "round-trip-hash")
        self.assertEqual(row[1], "us")
        self.assertEqual(row[2], '{"rule": "example"}')
        self.assertAlmostEqual(row[3], 1.2)
        self.assertAlmostEqual(row[4], 0.55)
        self.assertEqual(row[5], 42)
        self.assertAlmostEqual(row[6], -0.1)
        self.assertAlmostEqual(row[7], 0.3)
        self.assertAlmostEqual(row[8], 0.8)
        self.assertAlmostEqual(row[9], 0.2)
        self.assertTrue(row[10])

    def test_duplicate_hash_is_updated_not_duplicated(self):
        self._make_run(hypothesis_hash="dup-hash", sharpe_ratio=1.0, gate_passed=False)
        self._make_run(hypothesis_hash="dup-hash", sharpe_ratio=2.5, gate_passed=True)

        with _db_connection() as con:
            rows = con.execute(
                "SELECT sharpe_ratio, gate_passed FROM factory_runs WHERE hypothesis_hash = %s",
                ["dup-hash"],
            ).fetchall()

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][0], 2.5)
        self.assertTrue(rows[0][1])

    def test_count_factory_runs_reflects_upsert_not_insert(self):
        self._make_run(hypothesis_hash="count-hash", sharpe_ratio=1.0)
        before = count_factory_runs()
        # 同一ハッシュでの再保存はUPDATEであり、件数が増えてはならない。
        self._make_run(hypothesis_hash="count-hash", sharpe_ratio=1.5)
        after = count_factory_runs()
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
