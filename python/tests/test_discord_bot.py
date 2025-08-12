import os
import sys
import tempfile
import pandas as pd
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api.discord_bot import get_top10_diff_stocks_message

class TestGetTop10DiffStocksMessage(unittest.TestCase):
    def test_normal(self):
        df = pd.DataFrame({
            "market": ["us", "jp"],
            "symbol": ["AAPL", "SONY"],
            "current_price": [100, 200],
            "avg_pred_price": [110, 210],
            "diff_ratio": [0.1, 0.05],
            "model_count": [2, 1]
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            df.to_csv(csv_path, index=False)
            msg = get_top10_diff_stocks_message(csv_path)
            self.assertIn("AAPL", msg)
            self.assertIn("SONY", msg)
            self.assertIn("=== 差異割合上位10銘柄", msg)

    def test_not_found(self):
        msg = get_top10_diff_stocks_message("not_exist.csv")
        self.assertIn("結果CSVが見つかりませんでした", msg)

if __name__ == "__main__":
    unittest.main()
