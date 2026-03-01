"""df_to_string モジュールのユニットテスト"""

import unittest
import pandas as pd

from src.utils.df_to_string import df_to_pretty_string


class TestDfToPrettyString(unittest.TestCase):
    """df_to_pretty_string 関数のテスト"""

    def setUp(self):
        """テスト用のDataFrameを作成"""
        self.df = pd.DataFrame({
            "symbol": ["AAPL", "GOOG", "MSFT"],
            "price": [150.0, 2800.0, 300.0],
            "change": [0.05, -0.02, 0.01],
        })

    def test_returns_string(self):
        """戻り値がstr型であることを確認"""
        result = df_to_pretty_string(self.df)
        self.assertIsInstance(result, str)

    def test_contains_all_values(self):
        """全てのデータ値が出力に含まれることを確認"""
        result = df_to_pretty_string(self.df)
        self.assertIn("AAPL", result)
        self.assertIn("GOOG", result)
        self.assertIn("MSFT", result)

    def test_with_header(self):
        """ヘッダーが先頭に含まれることを確認"""
        header = "=== テストヘッダー ==="
        result = df_to_pretty_string(self.df, header=header)
        self.assertTrue(result.startswith(header))

    def test_without_header(self):
        """ヘッダーなしで正常に動作することを確認"""
        result = df_to_pretty_string(self.df)
        # ヘッダーがないのでデータ行のみ
        self.assertIn("AAPL", result)

    def test_column_filter(self):
        """columns指定時に指定したカラムのみ出力されることを確認"""
        result = df_to_pretty_string(self.df, columns=["symbol", "price"])
        self.assertIn("AAPL", result)
        self.assertIn("150", result)
        # change列は含まれない
        self.assertNotIn("0.05", result)

    def test_no_index(self):
        """出力にインデックスが含まれないことを確認"""
        result = df_to_pretty_string(self.df)
        # pandasのデフォルトインデックス (0, 1, 2) が行頭に表示されない
        lines = result.strip().split("\n")
        for line in lines[1:]:  # ヘッダー行を除く
            self.assertFalse(line.strip().startswith("0 "))

    def test_empty_dataframe(self):
        """空のDataFrameでもエラーにならないことを確認"""
        empty_df = pd.DataFrame(columns=["a", "b"])
        result = df_to_pretty_string(empty_df)
        self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()
