"""ユニットテスト: discord_utils"""

import unittest
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from src.reporting.discord.discord_text import split_text_chunks
from src.reporting.discord.discord_utils import (
    send_daily_order_completion,
    send_shap_notification,
    send_webhook_notification,
    send_webhook_text_chunked,
    send_weekly_report,
)


class TestSendDailyOrderCompletion(unittest.TestCase):
    @patch("src.reporting.discord.discord_utils.send_webhook_notification", return_value=True)
    def test_normal_completion_notification(self, mock_send):
        result = send_daily_order_completion(buy_orders=2, sell_orders=1, mode="paper")

        self.assertTrue(result)
        mock_send.assert_called_once()
        title, message = mock_send.call_args.args[:2]
        color = mock_send.call_args.kwargs["color"]

        self.assertEqual(title, "✅ 自動発注完了")
        self.assertIn("モード: paper", message)
        self.assertIn("買い注文: 2 件", message)
        self.assertIn("売り注文: 1 件", message)
        self.assertIn("JST", message)
        self.assertNotIn("停止理由:", message)
        self.assertEqual(color, 0x00BFFF)

    @patch("src.reporting.discord.discord_utils.send_webhook_notification", return_value=True)
    def test_stopped_completion_notification_contains_stop_details(self, mock_send):
        result = send_daily_order_completion(
            buy_orders=0,
            sell_orders=0,
            mode="paper",
            trading_stopped=True,
            stop_reason="日次損失上限に到達",
            daily_loss=25_000.0,
            daily_loss_limit=20_000.0,
        )

        self.assertTrue(result)
        mock_send.assert_called_once()
        title, message = mock_send.call_args.args[:2]
        color = mock_send.call_args.kwargs["color"]

        self.assertEqual(title, "⚠️ 自動発注停止")
        self.assertIn("停止理由: 日次損失上限に到達", message)
        self.assertIn("当日損失: 25000 円 / 上限: 20000 円", message)
        self.assertIn("JST", message)
        self.assertEqual(color, 0xFF9900)


class TestSendWebhookNotification(unittest.TestCase):
    @patch("src.reporting.discord.discord_utils._post_webhook")
    @patch(
        "src.reporting.discord.discord_utils.isoformat_jst",
        return_value="2026-04-06T09:30:45+09:00",
    )
    def test_embed_timestamp_uses_jst_isoformat(self, mock_isoformat, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = send_webhook_notification("title", "message")

        self.assertTrue(result)
        mock_isoformat.assert_called_once_with()
        payload = mock_post.call_args.kwargs["json_payload"]
        self.assertEqual(payload["embeds"][0]["timestamp"], "2026-04-06T09:30:45+09:00")


class TestChunkedTextHelpers(unittest.TestCase):
    def test_split_text_chunks_preserves_lines_when_possible(self):
        chunks = split_text_chunks("a\nb\nc", limit=3, preserve_lines=True)

        self.assertEqual(chunks, ["a\nb", "c"])

    @patch("src.reporting.discord.discord_utils.send_webhook_text", return_value=True)
    def test_send_webhook_text_chunked_sends_all_chunks(self, mock_send):
        result = send_webhook_text_chunked("12345", limit=2, preserve_lines=False)

        self.assertTrue(result)
        self.assertEqual([call.args[0] for call in mock_send.call_args_list], ["12", "34", "5"])


class TestSendShapNotification(unittest.TestCase):
    @patch("src.reporting.discord.discord_utils.send_webhook_text", return_value=True)
    def test_sends_top_and_bottom_sections(self, mock_send):
        shap_df = pd.DataFrame(
            {
                "feature": [f"feat_{i}" for i in range(1, 13)],
                "shap_mean": [float(13 - i) / 10 for i in range(1, 13)],
                "shap_rank": list(range(1, 13)),
            }
        )

        result = send_shap_notification("jp", "7203", "StockXGBoostModel", shap_df)

        self.assertTrue(result)
        self.assertGreaterEqual(mock_send.call_count, 1)
        sent_text = "\n".join(call.args[0] for call in mock_send.call_args_list)
        self.assertIn("SHAP特徴量寄与 [jp/7203] StockXGBoostModel", sent_text)
        self.assertIn("上位（寄与大）", sent_text)
        self.assertIn("下位（寄与小）", sent_text)
        self.assertIn("feat_1", sent_text)
        self.assertIn("feat_12", sent_text)

    @patch("src.reporting.discord.discord_utils.send_webhook_text", return_value=True)
    def test_returns_false_for_empty_dataframe(self, mock_send):
        result = send_shap_notification(
            "jp",
            "7203",
            "StockXGBoostModel",
            pd.DataFrame(columns=["feature", "shap_mean", "shap_rank"]),
        )

        self.assertFalse(result)
        mock_send.assert_not_called()


class TestSendWeeklyReport(unittest.TestCase):
    @patch("src.reporting.discord.discord_utils.send_webhook_text", return_value=True)
    def test_appends_paper_real_diff_section(self, mock_send):
        accuracy_df = pd.DataFrame(
            [
                {
                    "market": "jp",
                    "symbol": "7203",
                    "direction_accuracy": 0.55,
                    "mean_abs_error": 0.012,
                    "n_samples": 20,
                }
            ]
        )
        diff_summary = {
            "tracked_count": 4,
            "comparable_count": 2,
            "avg_paper_slippage": 0.001,
            "avg_real_slippage": 0.002,
            "avg_abs_price_diff": 3.5,
            "avg_abs_diff_ratio": 0.0035,
            "max_abs_price_diff": 7.0,
        }

        result = send_weekly_report(accuracy_df=accuracy_df, horizon=1, diff_summary=diff_summary)

        self.assertTrue(result)
        sent_text = "\n".join(call.args[0] for call in mock_send.call_args_list)
        self.assertIn("paper/real 乖離サマリー", sent_text)
        self.assertIn("tracked=4件, comparable=2件", sent_text)
        self.assertIn("平均価格差=3.500", sent_text)


if __name__ == "__main__":
    unittest.main()


# ──────────────────────────────────────────────────────────────
# pytest スタイルの追加テスト
# ──────────────────────────────────────────────────────────────


class TestPostWebhookErrorHandling:
    """_post_webhook エラーハンドリングのテスト"""

    @patch("src.reporting.discord.discord_utils._get_webhook_url")
    def test_returns_none_when_no_webhook_url(self, mock_url):
        """Webhook URL 未設定時は None が返ること"""
        from src.reporting.discord.discord_utils import _post_webhook

        mock_url.return_value = ""
        result = _post_webhook(json_payload={"content": "test"})
        assert result is None

    @patch("src.reporting.discord.discord_utils._get_webhook_url")
    @patch("src.reporting.discord.discord_utils.requests.post")
    def test_raises_on_http_error(self, mock_post, mock_url):
        """HTTP エラー時は例外が発生すること"""
        import requests

        from src.reporting.discord.discord_utils import _post_webhook

        mock_url.return_value = "https://discord.com/webhook"
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_post.return_value = mock_resp
        with pytest.raises(requests.exceptions.HTTPError):
            _post_webhook(json_payload={"content": "test"})


class TestSendWebhookTextChunkedExtra:
    """send_webhook_text_chunked の追加テスト"""

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    def test_sends_single_chunk_for_short_text(self, mock_send):
        """短いテキストは1チャンクで送信されること"""
        from src.reporting.discord.discord_utils import send_webhook_text_chunked

        mock_send.return_value = True
        result = send_webhook_text_chunked("Hello", limit=2000)
        assert result is True
        mock_send.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    def test_returns_false_if_any_chunk_fails(self, mock_send):
        """一部チャンクが失敗した場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_webhook_text_chunked

        # send_webhook_text が False を返すとき全体も False になること
        mock_send.return_value = False
        result = send_webhook_text_chunked("Hello\nWorld\nTest", limit=2000)
        assert result is False


class TestSendTextFileChunked:
    """send_text_file_chunked のテスト"""

    @patch("src.reporting.discord.discord_utils.send_webhook_text_chunked")
    def test_reads_and_sends_file(self, mock_send, tmp_path):
        """ファイルを読み込んで送信すること"""
        from src.reporting.discord.discord_utils import send_text_file_chunked

        mock_send.return_value = True
        f = tmp_path / "test.txt"
        f.write_text("Hello Discord", encoding="utf-8")
        result = send_text_file_chunked(str(f))
        assert result is True
        mock_send.assert_called_once()

    def test_returns_false_on_file_not_found(self):
        """ファイルが存在しない場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_text_file_chunked

        result = send_text_file_chunked("/nonexistent/path/file.txt")
        assert result is False


class TestSendStatusNotification:
    """send_status_notification のテスト"""

    @patch("src.reporting.discord.discord_utils.send_webhook_notification")
    def test_calls_webhook_notification_with_title(self, mock_notify):
        """spec のタイトルと行を結合して通知すること"""
        from src.reporting.discord.discord_notification_specs import NotificationSpec
        from src.reporting.discord.discord_utils import send_status_notification

        mock_notify.return_value = True
        spec = NotificationSpec(title="テストタイトル", color=0x00FF00)
        result = send_status_notification(spec, ["行1", "行2"])
        assert result is True
        mock_notify.assert_called_once()
        call_args = mock_notify.call_args
        assert call_args[0][0] == "テストタイトル"
        assert "行1" in call_args[0][1]


class TestSendWebhookNotificationExtra:
    """send_webhook_notification の追加テスト"""

    @patch("src.reporting.discord.rate_limiter.check_and_record", return_value=(True, None))
    @patch("src.reporting.discord.rate_limiter.apply_rate_limit")
    @patch("src.reporting.discord.discord_utils._post_webhook")
    @patch("src.reporting.discord.discord_utils.isoformat_jst")
    def test_returns_false_on_request_exception(self, mock_ts, mock_post, _rl, _ded):
        """RequestException 時は False が返ること"""
        import requests

        from src.reporting.discord.discord_utils import send_webhook_notification

        mock_ts.return_value = "2026-04-18T00:00:00+09:00"
        mock_post.side_effect = requests.exceptions.ConnectionError("接続エラー")
        result = send_webhook_notification("タイトル", "メッセージ")
        assert result is False

    @patch("src.reporting.discord.rate_limiter.check_and_record", return_value=(True, None))
    @patch("src.reporting.discord.rate_limiter.apply_rate_limit")
    @patch("src.reporting.discord.discord_utils._post_webhook")
    @patch("src.reporting.discord.discord_utils.isoformat_jst")
    def test_returns_false_when_response_is_none(self, mock_ts, mock_post, _rl, _ded):
        """レスポンスが None の場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_webhook_notification

        mock_ts.return_value = "2026-04-18T00:00:00+09:00"
        mock_post.return_value = None
        result = send_webhook_notification("タイトル", "メッセージ")
        assert result is False


# ──────────────────────────────────────────────────────────────
# 追加テストクラス（unittest.TestCase スタイル）
# ──────────────────────────────────────────────────────────────


class TestGetWebhookUrl(unittest.TestCase):
    """_get_webhook_url のテスト"""

    @patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test"})
    def test_returns_url_from_env(self):
        """環境変数から URL が返ること"""
        from src.reporting.discord.discord_utils import _get_webhook_url

        result = _get_webhook_url()
        self.assertEqual(result, "https://discord.com/api/webhooks/test")

    @patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": ""})
    def test_returns_none_when_empty_env(self):
        """環境変数が空文字の場合は None が返ること"""
        from src.reporting.discord.discord_utils import _get_webhook_url

        result = _get_webhook_url()
        self.assertIsNone(result)


class TestPostWebhook(unittest.TestCase):
    """_post_webhook のテスト"""

    @patch(
        "src.reporting.discord.discord_utils._get_webhook_url",
        return_value="https://webhook.url/test",
    )
    @patch("src.reporting.discord.discord_utils.requests.post")
    def test_posts_json_payload(self, mock_post, mock_url):
        """有効な URL がある場合に requests.post が呼ばれること"""
        from src.reporting.discord.discord_utils import _post_webhook

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        _post_webhook(json_payload={"content": "test"})
        mock_post.assert_called_once()

    @patch("src.reporting.discord.discord_utils._get_webhook_url", return_value=None)
    def test_returns_none_when_no_url(self, mock_url):
        """URL 未設定時は None が返ること"""
        from src.reporting.discord.discord_utils import _post_webhook

        result = _post_webhook(json_payload={"content": "test"})
        self.assertIsNone(result)


class TestSendWebhookFile(unittest.TestCase):
    """send_webhook_file のテスト"""

    @patch("src.reporting.discord.discord_utils._post_webhook")
    @patch("builtins.open", mock_open(read_data=b"test data"))
    @patch("src.reporting.discord.discord_utils.os.path.exists", return_value=True)
    def test_sends_file_when_exists(self, mock_exists, mock_post):
        """ファイルが存在する場合に _post_webhook が呼ばれ True が返ること"""
        from src.reporting.discord.discord_utils import send_webhook_file

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = send_webhook_file("/tmp/test.csv", "test comment")
        mock_post.assert_called_once()
        self.assertTrue(result)

    @patch("src.reporting.discord.discord_utils.os.path.exists", return_value=False)
    def test_skips_when_file_not_exists(self, mock_exists):
        """ファイルが存在しない場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_webhook_file

        result = send_webhook_file("/tmp/nonexistent.csv", "test")
        self.assertFalse(result)

    @patch("src.reporting.discord.discord_utils._post_webhook", return_value=None)
    @patch("builtins.open", mock_open(read_data=b"test data"))
    @patch("src.reporting.discord.discord_utils.os.path.exists", return_value=True)
    def test_returns_false_when_post_returns_none(self, mock_exists, mock_post):
        """_post_webhook が None を返す場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_webhook_file

        result = send_webhook_file("/tmp/test.csv", "test")
        self.assertFalse(result)


class TestSendDriftAlert(unittest.TestCase):
    """send_drift_alert のテスト"""

    @patch("src.reporting.discord.discord_utils.send_webhook_text", return_value=True)
    def test_sends_alert_when_accuracy_below_threshold(self, mock_send):
        """閾値以下の direction_accuracy がある場合に通知が送信されること"""
        from src.reporting.discord.discord_utils import send_drift_alert

        drift_df = pd.DataFrame(
            {
                "market": ["jp", "jp"],
                "symbol": ["7203", "9984"],
                "direction_accuracy": [0.3, 0.4],
                "mean_abs_error": [0.01, 0.02],
                "n_samples": [10, 20],
            }
        )
        result = send_drift_alert(drift_df, threshold=0.5)
        mock_send.assert_called_once()
        self.assertTrue(result)

    @patch("src.reporting.discord.discord_utils.send_webhook_text", return_value=True)
    def test_no_alert_when_accuracy_above_threshold(self, mock_send):
        """閾値を超える direction_accuracy のみの場合は通知が送信されないこと"""
        from src.reporting.discord.discord_utils import send_drift_alert

        drift_df = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "direction_accuracy": [0.8],
                "mean_abs_error": [0.01],
                "n_samples": [10],
            }
        )
        result = send_drift_alert(drift_df, threshold=0.5)
        mock_send.assert_not_called()
        self.assertFalse(result)

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    def test_returns_false_for_empty_df(self, mock_send):
        """空 DataFrame の場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_drift_alert

        result = send_drift_alert(pd.DataFrame(), threshold=0.5)
        mock_send.assert_not_called()
        self.assertFalse(result)

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    def test_returns_false_for_none(self, mock_send):
        """None を渡した場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_drift_alert

        result = send_drift_alert(None)
        mock_send.assert_not_called()
        self.assertFalse(result)


class TestSendWeeklyReportExtra(unittest.TestCase):
    """send_weekly_report の追加テスト"""

    @patch("src.reporting.discord.discord_utils.send_webhook_text", return_value=True)
    def test_sends_report_with_valid_accuracy_df(self, mock_send):
        """有効な accuracy_df がある場合にレポートが送信されること"""
        from src.reporting.discord.discord_utils import send_weekly_report

        accuracy_df = pd.DataFrame(
            {
                "market": ["jp"],
                "symbol": ["7203"],
                "direction_accuracy": [0.65],
                "mean_abs_error": [0.012],
                "n_samples": [20],
            }
        )
        # diff_summary を明示的に渡して DB アクセスを回避
        diff_summary = {"tracked_count": 0, "comparable_count": 0}
        result = send_weekly_report(accuracy_df=accuracy_df, diff_summary=diff_summary)
        self.assertTrue(result)
        mock_send.assert_called()

    @patch("src.utils.db.load_drift_summary")
    def test_returns_false_when_accuracy_df_none_and_db_empty(self, mock_load):
        """accuracy_df=None かつ DB が空の場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_weekly_report

        mock_load.return_value = pd.DataFrame()
        result = send_weekly_report(accuracy_df=None)
        self.assertFalse(result)
