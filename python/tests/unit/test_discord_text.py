"""ユニットテスト: discord_text"""

import unittest

from src.reporting.discord.discord_text import split_text_chunks


class TestSplitTextChunks(unittest.TestCase):
    def test_splits_long_line_without_preserving_lines(self):
        chunks = split_text_chunks("12345", limit=2, preserve_lines=False)

        self.assertEqual(chunks, ["12", "34", "5"])

    def test_preserves_lines_when_possible(self):
        chunks = split_text_chunks("aa\nbb\ncc", limit=5, preserve_lines=True)

        self.assertEqual(chunks, ["aa\nbb", "cc"])

    def test_short_text_returns_single_chunk(self):
        chunks = split_text_chunks("hello", limit=100)
        self.assertEqual(chunks, ["hello"])

    def test_line_exceeding_limit_is_split_recursively(self):
        long_line = "A" * 10
        chunks = split_text_chunks(long_line, limit=4, preserve_lines=True)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 4)
        self.assertEqual("".join(chunks), long_line)

    def test_mixed_normal_and_long_lines(self):
        text = "short\n" + "X" * 10 + "\nend"
        chunks = split_text_chunks(text, limit=6, preserve_lines=True)
        reconstructed = "\n".join(chunks)
        self.assertIn("short", reconstructed)
        self.assertIn("end", reconstructed)


if __name__ == "__main__":
    unittest.main()
