"""ユニットテスト: Discord の1通の上限を超える文を分ける・切り詰める（#810）"""

from unittest.mock import patch

import pandas as pd
import pytest
import requests

from src.reporting.discord import webhook_sender
from src.reporting.discord.notifications_drift import (
    send_accuracy_summary,
    send_drift_alert,
    send_drift_retrain_notification,
)
from src.reporting.discord.notifications_model import send_feature_suggestion_notification

_CONTENT_LIMIT = 2000
_DESCRIPTION_LIMIT = 4096
_FIELDS_LIMIT = 25
_EMBED_TOTAL_LIMIT = 6000
_FAKE_URL = "https://example.invalid/hook"


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """同じ文の抑止の記録をほかのテストに残さない。"""
    with patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    ):
        yield


def _summary_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": ["us"] * n,
            "symbol": [f"SYM{i:04d}" for i in range(n)],
            "direction_accuracy": [0.30 + (i % 50) / 100 for i in range(n)],
            "mean_abs_error": [0.0123] * n,
            "n_samples": [20] * n,
        }
    )


def _contents(mock_post) -> list[str]:
    return [c.kwargs["json_payload"]["content"] for c in mock_post.call_args_list]


def _embed_total(embed: dict) -> int:
    total = len(embed.get("title", "")) + len(embed.get("description", ""))
    for field in embed.get("fields", []):
        total += len(field["name"]) + len(field["value"])
    total += len(embed.get("footer", {}).get("text", ""))
    return total


class TestTextWithinLimit:
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_accuracy_summary_of_707_symbols_fits(self, mock_post):
        """精度サマリーは銘柄が多くても上限の内で、最も悪い銘柄を載せ、省いた数を書く"""
        assert send_accuracy_summary(_summary_df(707), horizon=1) is True
        contents = _contents(mock_post)
        assert contents
        assert all(len(c) <= _CONTENT_LIMIT for c in contents)
        assert "SYM0000" in contents[0]
        assert "省略" in contents[-1]

    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_drift_alert_of_many_symbols_fits(self, mock_post):
        assert send_drift_alert(_summary_df(700), horizon=1, threshold=0.45) is True
        contents = _contents(mock_post)
        assert contents
        assert all(len(c) <= _CONTENT_LIMIT for c in contents)

    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_drift_retrain_notification_of_many_symbols_fits(self, mock_post):
        triggered = [
            {
                "market": "jp",
                "symbol": f"{1000 + i}.T",
                "mean_abs_error": 0.05,
                "direction_accuracy": 0.4,
            }
            for i in range(200)
        ]
        assert send_drift_retrain_notification(triggered, 0.02, 0.45) is True
        contents = _contents(mock_post)
        assert contents
        assert all(len(c) <= _CONTENT_LIMIT for c in contents)

    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_long_plain_text_is_split(self, mock_post):
        """上限を超える文は、行を保って複数の通に分けて全部送る"""
        lines = [f"line-{i:04d} " + "x" * 40 for i in range(120)]
        assert webhook_sender.send_webhook_text("\n".join(lines)) is True
        contents = _contents(mock_post)
        assert len(contents) > 1
        assert all(len(c) <= _CONTENT_LIMIT for c in contents)
        assert "\n".join(contents) == "\n".join(lines)


class TestEmbedWithinLimit:
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_feature_suggestion_of_50_symbols_fits(self, mock_post):
        """ドリフトのチェックの後の特徴量除外提案（embed）は上限の内に収まる"""
        candidates = pd.DataFrame(
            {
                "feature": [f"feature_name_{j:02d}" for j in range(10)],
                "importance_mean": [-0.000123] * 10,
                "importance_rank": list(range(40, 50)),
            }
        )
        suggestions = [
            {"market": "us", "symbol": f"SYM{i:02d}", "candidates": candidates} for i in range(50)
        ]
        assert send_feature_suggestion_notification(suggestions) is True
        embed = mock_post.call_args.kwargs["json_payload"]["embeds"][0]
        assert len(embed["description"]) <= _DESCRIPTION_LIMIT
        assert _embed_total(embed) <= _EMBED_TOTAL_LIMIT
        assert "省略" in embed["description"]

    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_fields_are_capped(self, mock_post):
        fields = [{"name": f"name{i}", "value": "v" * 2000, "inline": True} for i in range(40)]
        assert webhook_sender.send_webhook_notification("t" * 300, "d", fields=fields) is True
        embed = mock_post.call_args.kwargs["json_payload"]["embeds"][0]
        assert len(embed["title"]) <= 256
        assert len(embed["fields"]) <= _FIELDS_LIMIT
        assert all(len(f["value"]) <= 1024 for f in embed["fields"])
        assert _embed_total(embed) <= _EMBED_TOTAL_LIMIT


def _bad_request() -> requests.Response:
    response = requests.Response()
    response.status_code = 400
    response.reason = "Bad Request"
    response.url = _FAKE_URL
    response._content = b'{"message": "Invalid Form Body", "code": 50035}'
    return response


class TestFailureLogsResponseBody:
    @patch("src.reporting.discord.webhook_sender._get_webhook_url", return_value=_FAKE_URL)
    @patch("src.reporting.discord.webhook_sender.requests.post")
    def test_text_failure_logs_body_without_url(self, mock_post, _mock_url):
        mock_post.return_value = _bad_request()
        with patch.object(webhook_sender.logger, "error") as mock_error:
            assert webhook_sender.send_webhook_text("hello") is False
        logged = " ".join(str(a) for c in mock_error.call_args_list for a in c.args)
        assert "Invalid Form Body" in logged
        assert "400" in logged
        assert _FAKE_URL not in logged

    @patch("src.reporting.discord.webhook_sender._get_webhook_url", return_value=_FAKE_URL)
    @patch("src.reporting.discord.webhook_sender.requests.post")
    def test_embed_failure_logs_body_without_url(self, mock_post, _mock_url):
        mock_post.return_value = _bad_request()
        with patch.object(webhook_sender.logger, "error") as mock_error:
            assert webhook_sender.send_webhook_notification("title", "message") is False
        logged = " ".join(str(a) for c in mock_error.call_args_list for a in c.args)
        assert "Invalid Form Body" in logged
        assert _FAKE_URL not in logged
