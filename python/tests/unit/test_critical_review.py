"""ユニットテスト: バックテスト批判的レビュー（src/backtest/critical_review.py）

LLM は InMemoryTextReviewPort を注入し、API 呼び出しは行わない。
レポート書き込みは tmp ディレクトリへ向ける。
"""

import json
import unittest
from unittest.mock import patch

from src.backtest import critical_review
from src.infrastructure.in_memory import InMemoryTextReviewPort


def _port(findings):
    """構造化出力 findings を返す InMemoryTextReviewPort を返す。"""
    return InMemoryTextReviewPort([json.dumps({"findings": findings})])


_SAMPLE = [
    {
        "title": "スリッページがデフォルト0",
        "severity": "high",
        "category": "コスト過小評価",
        "rationale": "約定コストを無視しており純損益が楽観的になる。",
        "suggestion": "現実的なスリッページを既定値に設定する。",
    }
]


class TestRunBacktestReview(unittest.TestCase):
    @patch("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", False)
    def test_disabled_returns_empty(self):
        port = _port(_SAMPLE)
        self.assertEqual(critical_review.run_backtest_review(review_port=port), [])
        self.assertEqual(port.calls, [])

    @patch("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", True)
    @patch("src.backtest.critical_review._gather_review_context", return_value="ctx")
    def test_dry_run_does_not_write(self, _ctx):
        with patch("src.backtest.critical_review._write_finding_report") as mock_write:
            findings = critical_review.run_backtest_review(review_port=_port(_SAMPLE), dry_run=True)
        self.assertEqual(len(findings), 1)
        mock_write.assert_not_called()

    @patch("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", True)
    @patch("src.backtest.critical_review._gather_review_context", return_value="ctx")
    def test_writes_reports(self, _ctx):
        port = _port(_SAMPLE)
        with patch("src.backtest.critical_review._write_finding_report") as mock_write:
            mock_write.return_value = "path.json"
            findings = critical_review.run_backtest_review(review_port=port, dry_run=False)
        self.assertEqual(len(findings), 1)
        mock_write.assert_called_once()
        # 構造化出力（JSON Schema）を要求していること
        self.assertEqual(port.calls[0]["schema"], critical_review._FINDINGS_SCHEMA)
        self.assertEqual(port.calls[0]["user"], "ctx")

    @patch("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", True)
    @patch("src.backtest.critical_review._gather_review_context", return_value="ctx")
    def test_api_error_returns_empty(self, _ctx):
        port = InMemoryTextReviewPort(error=RuntimeError("API down"))
        self.assertEqual(critical_review.run_backtest_review(review_port=port), [])

    @patch("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", True)
    @patch("src.backtest.critical_review._gather_review_context", return_value="ctx")
    def test_no_findings_returns_empty(self, _ctx):
        self.assertEqual(critical_review.run_backtest_review(review_port=_port([])), [])


class TestReportWriting(unittest.TestCase):
    def test_finding_hash_is_stable_and_namespaced(self):
        h1 = critical_review._finding_hash("cat", "title")
        h2 = critical_review._finding_hash("cat", "title")
        self.assertEqual(h1, h2)
        self.assertTrue(h1.startswith("review-"))
        # カテゴリ/タイトルが変われば別ハッシュ
        self.assertNotEqual(h1, critical_review._finding_hash("cat", "other"))

    def test_write_finding_report_matches_intake_schema(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.critical_review._reports_dir", return_value=tmp):
                path = critical_review._write_finding_report(_SAMPLE[0])
            self.assertTrue(path.endswith(".json"))
            with open(path, encoding="utf-8") as f:
                report = json.load(f)

        self.assertEqual(report["schema_version"], 1)
        self.assertIn("[factory:review-", report["issue_title"])
        self.assertIn("BTレビュー", report["issue_title"])
        self.assertEqual(report["labels"], ["backtest", "P2"])  # severity=high

    def test_write_finding_report_skips_empty_title(self):
        result = critical_review._write_finding_report({"title": "", "category": "c"})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
