import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.utils.data_path_utils as path_utils  # noqa: E402
import src.utils.db as db_module  # noqa: E402
from src.api.discord_bot import get_top10_diff_stocks_message  # noqa: E402


class TestGetTop10DiffStocksMessage(unittest.TestCase):
    def setUp(self):
        """テスト用一時DBを設定"""
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_discord.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal_path = self.tmp_db + ".wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    def test_normal(self):
        from src.domain.types import PredictionResult
        from src.utils.db import save_prediction_results

        results = [
            PredictionResult(
                market="us",
                symbol="AAPL",
                current_price=100.0,
                avg_pred_price=110.0,
                diff_ratio=0.1,
                model_count=2,
            ),
            PredictionResult(
                market="us",
                symbol="SONY",
                current_price=200.0,
                avg_pred_price=210.0,
                diff_ratio=0.05,
                model_count=1,
            ),
        ]
        save_prediction_results("20260228_120000", results)
        msg = get_top10_diff_stocks_message("us", "top10", "20260228_120000")
        print(msg)
        self.assertIn("AAPL", msg)
        self.assertIn("SONY", msg)

    def test_not_found(self):
        msg = get_top10_diff_stocks_message("us", "top10", "19700101_000000")
        print(msg)
        self.assertEqual(msg, "")


if __name__ == "__main__":
    unittest.main()
