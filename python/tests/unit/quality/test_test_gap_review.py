"""ユニットテスト: テスト穴埋めボット（src/quality/test_gap_review.py）

TextReviewPort はモックで差し替え、実 LLM 呼び出しは行わない。
運用が cli でもテストは LLM_BACKEND="sdk" に固定し、ambient env 依存で
実 claude CLI を起動しないようにする（#516 の教訓）。
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.quality import test_gap_review
from src.quality.types import CoverageTarget


def _mock_port(suggestions):
    """suggestions を構造化出力で返す TextReviewPort モックを返す。"""
    port = MagicMock()
    port.complete.return_value = json.dumps({"suggestions": suggestions})
    return port


_SAMPLE = [
    {
        "target_path": "src/foo/bar.py",
        "uncovered_summary": "例外経路が未到達",
        "test_outline": "- 不正入力で ValueError を送出するケースを追加",
        "priority": "high",
    }
]

_TARGETS = [CoverageTarget(path="src/foo/bar.py", percent_covered=50.0, missing_lines=[10, 11])]


@patch("src.infrastructure.llm.factory.LLM_BACKEND", "sdk")
@patch("src.quality.test_gap_review.TEST_GAP_ENABLED", True)
@patch("src.quality.test_gap_review._gather_targets_context", return_value="ctx")
@patch("src.quality.test_gap_review._select_targets", return_value=_TARGETS)
@patch("src.quality.test_gap_review._load_coverage", return_value={"files": {}})
class TestRunTestGap(unittest.TestCase):
    @patch("src.infrastructure.llm.factory.get_text_review_port")
    def test_dry_run_does_not_write(self, mock_get_port, *_):
        mock_get_port.return_value = _mock_port(_SAMPLE)
        with patch("src.quality.test_gap_review._write_suggestion_report") as mock_write:
            suggestions = test_gap_review.run_test_gap_review(dry_run=True)
        self.assertEqual(len(suggestions), 1)
        mock_write.assert_not_called()

    @patch("src.infrastructure.llm.factory.get_text_review_port")
    def test_writes_reports(self, mock_get_port, *_):
        port = _mock_port(_SAMPLE)
        mock_get_port.return_value = port
        with patch("src.quality.test_gap_review._write_suggestion_report") as mock_write:
            mock_write.return_value = "path.json"
            suggestions = test_gap_review.run_test_gap_review(dry_run=False)
        self.assertEqual(len(suggestions), 1)
        mock_write.assert_called_once()
        # 構造化スキーマを要求していること
        _, kwargs = port.complete.call_args
        self.assertIsNotNone(kwargs.get("schema"))

    @patch("src.infrastructure.llm.factory.get_text_review_port")
    def test_port_error_returns_empty(self, mock_get_port, *_):
        port = MagicMock()
        port.complete.side_effect = RuntimeError("CLI down")
        mock_get_port.return_value = port
        self.assertEqual(test_gap_review.run_test_gap_review(), [])

    @patch("src.infrastructure.llm.factory.get_text_review_port")
    def test_no_suggestions_returns_empty(self, mock_get_port, *_):
        mock_get_port.return_value = _mock_port([])
        self.assertEqual(test_gap_review.run_test_gap_review(), [])


@patch("src.quality.test_gap_review.TEST_GAP_ENABLED", False)
class TestDisabled(unittest.TestCase):
    def test_disabled_returns_empty(self):
        self.assertEqual(test_gap_review.run_test_gap_review(), [])


@patch("src.quality.test_gap_review.TEST_GAP_ENABLED", True)
class TestNoTargets(unittest.TestCase):
    @patch("src.quality.test_gap_review._load_coverage", return_value={"files": {}})
    def test_empty_coverage_returns_empty(self, _load):
        # 対象なし → port を呼ばず空リスト
        self.assertEqual(test_gap_review.run_test_gap_review(), [])

    def test_bad_coverage_path_returns_empty(self):
        self.assertEqual(
            test_gap_review.run_test_gap_review(coverage_json="/no/such/coverage.json"), []
        )


class TestSelectTargets(unittest.TestCase):
    @staticmethod
    def _cov(percent, missing):
        return {"summary": {"percent_covered": percent}, "missing_lines": missing}

    def test_filters_and_sorts_by_missing_count(self):
        data = {
            "files": {
                "a.py": self._cov(50.0, [1, 2, 3]),
                "b.py": self._cov(90.0, [1]),  # 閾値以上 → 除外
                "c.py": self._cov(60.0, [1, 2, 3, 4, 5]),
                "d.py": self._cov(70.0, []),  # 未カバーなし → 除外
            }
        }
        targets = test_gap_review._select_targets(data)
        self.assertEqual([t.path for t in targets], ["c.py", "a.py"])

    def test_respects_max_files(self):
        files = {f"f{i}.py": self._cov(10.0, list(range(i + 1))) for i in range(10)}
        with patch("src.quality.test_gap_review.TEST_GAP_MAX_FILES", 3):
            targets = test_gap_review._select_targets({"files": files})
        self.assertEqual(len(targets), 3)


class TestContextBudget(unittest.TestCase):
    @patch("src.quality.test_gap_review.TEST_GAP_CONTEXT_CHAR_BUDGET", 300)
    @patch("src.quality.test_gap_review._read_source_excerpt", return_value="x" * 200)
    def test_budget_truncates_low_priority_targets(self, _excerpt):
        targets = [CoverageTarget(f"f{i}.py", 10.0, [1, 2, 3]) for i in range(5)]
        ctx = test_gap_review._gather_targets_context(targets)
        # 先頭（最重要）は必ず入り、予算超過で末尾は落ちる
        self.assertIn("f0.py", ctx)
        self.assertNotIn("f4.py", ctx)

    def test_read_source_excerpt_marks_missing_lines(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write("\n".join(f"line{i}" for i in range(1, 11)))
            name = f.name
        try:
            target = CoverageTarget(path=name, percent_covered=10.0, missing_lines=[5])
            excerpt = test_gap_review._read_source_excerpt(target, window=1)
        finally:
            os.unlink(name)
        self.assertIn(">>", excerpt)
        self.assertIn("line5", excerpt)  # 未カバー行
        self.assertIn("line4", excerpt)  # 前後コンテキスト


class TestReportWriting(unittest.TestCase):
    def test_finding_hash_is_stable_and_namespaced(self):
        h1 = test_gap_review._finding_hash("src/foo.py")
        h2 = test_gap_review._finding_hash("src/foo.py")
        self.assertEqual(h1, h2)
        self.assertTrue(h1.startswith("test-gap-"))
        self.assertNotEqual(h1, test_gap_review._finding_hash("src/bar.py"))

    def test_write_suggestion_report_matches_intake_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.quality.test_gap_review._reports_dir", return_value=tmp):
                path = test_gap_review._write_suggestion_report(_SAMPLE[0])
            self.assertTrue(path.endswith(".json"))
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        self.assertEqual(report["schema_version"], 1)
        self.assertIn("[factory:test-gap-", report["issue_title"])
        self.assertIn("テスト穴埋め", report["issue_title"])
        self.assertEqual(report["labels"], ["test", "P2"])  # priority=high

    def test_write_suggestion_report_skips_empty_target(self):
        result = test_gap_review._write_suggestion_report({"target_path": "", "priority": "low"})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
