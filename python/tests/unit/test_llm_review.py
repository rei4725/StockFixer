"""ユニットテスト: Claude 週次レビュー講評生成（src/reporting/llm_review.py）

anthropic クライアントは MagicMock で差し替え、API 呼び出しは行わない。
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.reporting import llm_review


def _make_accuracy_df():
    return pd.DataFrame(
        [
            {
                "market": "jp",
                "symbol": "7203",
                "direction_accuracy": 0.42,
                "mean_abs_error": 0.012,
                "n_samples": 30,
            },
            {
                "market": "jp",
                "symbol": "6758",
                "direction_accuracy": 0.61,
                "mean_abs_error": 0.008,
                "n_samples": 30,
            },
        ]
    )


def _make_diff_summary():
    return {
        "tracked_count": 12,
        "comparable_count": 8,
        "avg_paper_slippage": 0.001,
        "avg_real_slippage": 0.002,
        "avg_abs_price_diff": 1.5,
        "avg_abs_diff_ratio": 0.0015,
        "max_abs_price_diff": 5.0,
    }


def _mock_anthropic(text="(1)総評 (2)懸念 (3)推奨"):
    """anthropic.Anthropic を差し替えるモックを返す。"""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response = MagicMock()
    response.content = [text_block]
    client = MagicMock()
    client.messages.create.return_value = response
    anthropic_module = MagicMock()
    anthropic_module.Anthropic.return_value = client
    return anthropic_module, client


# 環境変数 LLM_BACKEND に依存させず、anthropic モックが効く SDK 経路に固定する
# （cli だと factory が実 claude CLI を起動してしまう）
@patch("src.infrastructure.llm.factory.LLM_BACKEND", "sdk")
class TestGenerateWeeklyReview(unittest.TestCase):
    @patch("src.reporting.llm_review.LLM_REVIEW_ENABLED", False)
    def test_disabled_returns_none(self):
        result = llm_review.generate_weekly_review(_make_accuracy_df())
        self.assertIsNone(result)

    @patch("src.reporting.llm_review.LLM_REVIEW_ENABLED", True)
    def test_empty_df_returns_none(self):
        result = llm_review.generate_weekly_review(pd.DataFrame())
        self.assertIsNone(result)

    @patch("src.reporting.llm_review.LLM_REVIEW_ENABLED", True)
    def test_none_df_returns_none(self):
        result = llm_review.generate_weekly_review(None)
        self.assertIsNone(result)

    @patch("src.reporting.llm_review.LLM_REVIEW_ENABLED", True)
    def test_enabled_returns_review_text(self):
        anthropic_module, client = _mock_anthropic("週次の総評テキスト")
        with patch.dict("sys.modules", {"anthropic": anthropic_module}):
            result = llm_review.generate_weekly_review(
                _make_accuracy_df(), diff_summary=_make_diff_summary(), horizon=1
            )
        self.assertEqual(result, "週次の総評テキスト")
        # モデル指定とプロンプト内容が渡っていること
        _, kwargs = client.messages.create.call_args
        self.assertIn("model", kwargs)
        self.assertIn("7203", kwargs["messages"][0]["content"])

    @patch("src.reporting.llm_review.LLM_REVIEW_ENABLED", True)
    def test_api_error_returns_none(self):
        anthropic_module = MagicMock()
        anthropic_module.Anthropic.side_effect = RuntimeError("API down")
        with patch.dict("sys.modules", {"anthropic": anthropic_module}):
            result = llm_review.generate_weekly_review(_make_accuracy_df())
        self.assertIsNone(result)

    @patch("src.reporting.llm_review.LLM_REVIEW_ENABLED", True)
    def test_empty_response_returns_none(self):
        anthropic_module, _ = _mock_anthropic(text="   ")
        with patch.dict("sys.modules", {"anthropic": anthropic_module}):
            result = llm_review.generate_weekly_review(_make_accuracy_df())
        self.assertIsNone(result)

    def test_build_metrics_digest_includes_key_figures(self):
        digest = llm_review._build_metrics_digest(
            _make_accuracy_df(), _make_diff_summary(), horizon=1
        )
        self.assertIn("全体平均方向正解率", digest)
        self.assertIn("7203", digest)
        self.assertIn("paper/real 乖離", digest)


class TestSendWeeklyReportLlmSection(unittest.TestCase):
    @patch("src.reporting.discord.notifications_report.send_webhook_text_chunked")
    def test_llm_review_section_appended(self, mock_send):
        mock_send.return_value = True
        from src.reporting.discord.notifications_report import send_weekly_report

        with patch(
            "src.utils.db.load_weekly_accuracy_snapshots",
            return_value=pd.DataFrame(),
        ):
            send_weekly_report(
                accuracy_df=_make_accuracy_df(),
                horizon=1,
                diff_summary={"tracked_count": 0},
                llm_review="これはClaude講評です",
            )

        sent_text = mock_send.call_args[0][0]
        self.assertIn("🧠 Claude 講評", sent_text)
        self.assertIn("これはClaude講評です", sent_text)

    @patch("src.reporting.discord.notifications_report.send_webhook_text_chunked")
    def test_no_llm_section_when_none(self, mock_send):
        mock_send.return_value = True
        from src.reporting.discord.notifications_report import send_weekly_report

        with patch(
            "src.utils.db.load_weekly_accuracy_snapshots",
            return_value=pd.DataFrame(),
        ):
            send_weekly_report(
                accuracy_df=_make_accuracy_df(),
                horizon=1,
                diff_summary={"tracked_count": 0},
                llm_review=None,
            )

        sent_text = mock_send.call_args[0][0]
        self.assertNotIn("Claude 講評", sent_text)


if __name__ == "__main__":
    unittest.main()
