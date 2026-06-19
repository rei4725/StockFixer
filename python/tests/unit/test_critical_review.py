"""ユニットテスト: バックテスト批判的レビュー（src/backtest/critical_review.py）

anthropic クライアントは MagicMock で差し替え、API 呼び出しは行わない。
レポート書き込みは tmp ディレクトリへ向ける。
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.backtest import critical_review


def _mock_anthropic(findings):
    """構造化出力 findings を返す anthropic モジュールモックを返す。"""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps({"findings": findings})
    response = MagicMock()
    response.content = [text_block]
    client = MagicMock()
    client.messages.create.return_value = response
    module = MagicMock()
    module.Anthropic.return_value = client
    return module, client


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
        self.assertEqual(critical_review.run_backtest_review(), [])

    @patch("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", True)
    @patch("src.backtest.critical_review._gather_review_context", return_value="ctx")
    def test_dry_run_does_not_write(self, _ctx):
        module, _client = _mock_anthropic(_SAMPLE)
        with patch.dict("sys.modules", {"anthropic": module}):
            with patch("src.backtest.critical_review._write_finding_report") as mock_write:
                findings = critical_review.run_backtest_review(dry_run=True)
        self.assertEqual(len(findings), 1)
        mock_write.assert_not_called()

    @patch("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", True)
    @patch("src.backtest.critical_review._gather_review_context", return_value="ctx")
    def test_writes_reports(self, _ctx):
        module, client = _mock_anthropic(_SAMPLE)
        with patch.dict("sys.modules", {"anthropic": module}):
            with patch("src.backtest.critical_review._write_finding_report") as mock_write:
                mock_write.return_value = "path.json"
                findings = critical_review.run_backtest_review(dry_run=False)
        self.assertEqual(len(findings), 1)
        mock_write.assert_called_once()
        # 構造化出力（json_schema）を要求していること
        _, kwargs = client.messages.create.call_args
        self.assertIn("output_config", kwargs)

    @patch("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", True)
    @patch("src.backtest.critical_review._gather_review_context", return_value="ctx")
    def test_api_error_returns_empty(self, _ctx):
        module = MagicMock()
        module.Anthropic.side_effect = RuntimeError("API down")
        with patch.dict("sys.modules", {"anthropic": module}):
            self.assertEqual(critical_review.run_backtest_review(), [])

    @patch("src.backtest.critical_review.BACKTEST_REVIEW_ENABLED", True)
    @patch("src.backtest.critical_review._gather_review_context", return_value="ctx")
    def test_no_findings_returns_empty(self, _ctx):
        module, _client = _mock_anthropic([])
        with patch.dict("sys.modules", {"anthropic": module}):
            self.assertEqual(critical_review.run_backtest_review(), [])


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
