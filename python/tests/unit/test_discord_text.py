"""ユニットテスト: discord_text"""

import unittest

from src.api.discord_text import split_text_chunks


class TestSplitTextChunks(unittest.TestCase):
    def test_splits_long_line_without_preserving_lines(self):
        chunks = split_text_chunks("12345", limit=2, preserve_lines=False)

        self.assertEqual(chunks, ["12", "34", "5"])

    def test_preserves_lines_when_possible(self):
        chunks = split_text_chunks("aa\nbb\ncc", limit=5, preserve_lines=True)

        self.assertEqual(chunks, ["aa\nbb", "cc"])


if __name__ == "__main__":
    unittest.main()
