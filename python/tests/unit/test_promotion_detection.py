import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.backtest.promotion_detection import (
    extract_closing_issue_numbers,
    extract_factory_hash,
    load_gate_baseline,
)


class TestExtractClosingIssueNumbers(unittest.TestCase):
    def test_closes_keyword(self):
        self.assertEqual(extract_closing_issue_numbers("Closes #564"), [564])

    def test_fixes_keyword_case_insensitive(self):
        self.assertEqual(extract_closing_issue_numbers("fixes #12"), [12])

    def test_resolved_keyword(self):
        self.assertEqual(extract_closing_issue_numbers("This resolved #99 finally"), [99])

    def test_multiple_keywords(self):
        self.assertEqual(
            sorted(extract_closing_issue_numbers("Closes #1\n\nAlso fixes #2")), [1, 2]
        )

    def test_no_keyword_returns_empty(self):
        self.assertEqual(extract_closing_issue_numbers("See #564 for context"), [])

    def test_empty_body_returns_empty(self):
        self.assertEqual(extract_closing_issue_numbers(""), [])
        self.assertEqual(extract_closing_issue_numbers(None), [])


class TestExtractFactoryHash(unittest.TestCase):
    def test_extracts_hash_from_marker(self):
        title = "[factory:fb44f0011174] AND合成ルール (jp)"
        self.assertEqual(extract_factory_hash(title), "fb44f0011174")

    def test_no_marker_returns_none(self):
        self.assertIsNone(extract_factory_hash("普通のタイトル"))

    def test_empty_title_returns_none(self):
        self.assertIsNone(extract_factory_hash(""))
        self.assertIsNone(extract_factory_hash(None))


class TestLoadGateBaseline(unittest.TestCase):
    def test_reads_champion_sharpe_from_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = os.path.join(tmp_dir, "factory", "reports")
            os.makedirs(reports_dir)
            report_path = os.path.join(reports_dir, "abc123.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({"gate": {"champion_sharpe": 1.42}}, f)

            with patch("src.backtest.promotion_detection.get_results_dir", return_value=tmp_dir):
                self.assertAlmostEqual(load_gate_baseline("abc123"), 1.42)

    def test_missing_report_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.backtest.promotion_detection.get_results_dir", return_value=tmp_dir):
                self.assertIsNone(load_gate_baseline("does-not-exist"))
