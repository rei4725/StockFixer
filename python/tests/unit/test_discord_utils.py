"""ユニットテスト: discord_utils"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from src.reporting.discord.discord_text import split_text_chunks
from src.reporting.discord.discord_utils import (
    send_backup_completion,
    send_daily_order_completion,
    send_daily_pipeline_error,
    send_daily_settle_completion,
    send_db_maintenance_completion,
    send_hit_rate_drift_alert,
    send_monthly_report_notification,
    send_optimization_completion,
    send_shadow_evaluation_notification,
    send_shap_notification,
    send_walk_forward_report_completion,
    send_webhook_notification,
    send_webhook_text_chunked,
    send_weekly_report,
    send_weekly_training_completion,
)


def _embed_from_post(mock_post) -> dict:
    """_post_webhook モックから最初の embed を取り出す。"""
    payload = mock_post.call_args.kwargs["json_payload"]
    return payload["embeds"][0]


def _fields_map(embed: dict) -> dict:
    return {f["name"]: f["value"] for f in embed["fields"]}


class TestSendDailyOrderCompletion(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_normal_completion_notification(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_daily_order_completion(buy_orders=2, sell_orders=1, mode="paper")

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "✅ 自動発注完了")
        self.assertEqual(embed["color"], 0x00BFFF)
        fields = _fields_map(embed)
        self.assertIn("JST", fields["🕐 時刻"])
        self.assertEqual(fields["⚙️ モード"], "paper")
        self.assertEqual(fields["🟢 買い注文"], "2 件")
        self.assertEqual(fields["🔴 売り注文"], "1 件")
        self.assertNotIn("⛔ 停止理由", fields)

    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_stopped_completion_notification_contains_stop_details(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

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
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "⚠️ 自動発注停止")
        self.assertEqual(embed["color"], 0xFF9900)
        fields = _fields_map(embed)
        self.assertEqual(fields["⛔ 停止理由"], "日次損失上限に到達")
        self.assertEqual(fields["💰 当日損失"], "25,000 円 / 上限: 20,000 円")


class TestSendDailyPipelineError(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_sends_error_as_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_daily_pipeline_error("データ取得に失敗")

        self.assertTrue(result)
        fields = _fields_map(_embed_from_post(mock_post))
        self.assertIn("JST", fields["🕐 時刻"])
        self.assertEqual(fields["❌ エラー"], "データ取得に失敗")


class TestSendDailySettleCompletion(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_sends_settled_count_as_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_daily_settle_completion(settled_count=1234)

        self.assertTrue(result)
        fields = _fields_map(_embed_from_post(mock_post))
        self.assertIn("JST", fields["🕐 時刻"])
        self.assertEqual(fields["📊 約定件数"], "1,234 件")


class TestSendWeeklyTrainingCompletion(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_sends_models_as_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_weekly_training_completion(["model_a", "model_b"])

        self.assertTrue(result)
        fields = _fields_map(_embed_from_post(mock_post))
        self.assertIn("JST", fields["🕐 時刻"])
        self.assertEqual(fields["📦 学習モデル数"], "2 件")
        self.assertIn("model_a", fields["🏷 学習済みモデル"])
        self.assertIn("model_b", fields["🏷 学習済みモデル"])

    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_empty_models_uses_placeholder(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_weekly_training_completion([])

        self.assertTrue(result)
        fields = _fields_map(_embed_from_post(mock_post))
        self.assertEqual(fields["📦 学習モデル数"], "0 件")
        self.assertEqual(fields["🏷 学習済みモデル"], "なし")


class TestSendOptimizationCompletion(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_sends_success_failed_as_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_optimization_completion(success=8, failed=2)

        self.assertTrue(result)
        fields = _fields_map(_embed_from_post(mock_post))
        self.assertIn("JST", fields["🕐 時刻"])
        self.assertEqual(fields["✅ 成功"], "8 銘柄")
        self.assertEqual(fields["⚠️ 失敗"], "2 銘柄")

    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_all_success_uses_check_icon(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_optimization_completion(success=10, failed=0)

        self.assertTrue(result)
        fields = _fields_map(_embed_from_post(mock_post))
        self.assertEqual(fields["✅ 失敗"], "0 銘柄")


class TestSendWalkForwardReportCompletion(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_sends_summary_as_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_walk_forward_report_completion(
            {"success": 5, "failed": 1, "total": 6, "markdown_path": None, "previous_path": None}
        )

        self.assertTrue(result)
        fields = _fields_map(_embed_from_post(mock_post))
        self.assertIn("JST", fields["🕐 時刻"])
        self.assertEqual(fields["✅ 成功"], "5 銘柄")
        self.assertEqual(fields["❌ 失敗"], "1 銘柄")
        self.assertEqual(fields["📊 合計"], "6 銘柄")
        self.assertEqual(fields["📄 前回比較"], "なし（初回実行）")


class TestSendDbMaintenanceCompletion(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_success_sends_sizes_as_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_db_maintenance_completion(
            elapsed_seconds=12.3, size_before_mb=120.5, size_after_mb=100.25
        )

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "✅ DB メンテナンス完了")
        fields = _fields_map(embed)
        self.assertIn("JST", fields["🕐 時刻"])
        self.assertEqual(fields["⏱ 処理時間"], "12.3 秒")
        self.assertEqual(fields["💾 DBサイズ"], "120.50 MB → 100.25 MB (-20.25 MB)")

    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_error_sends_error_field(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_db_maintenance_completion(
            elapsed_seconds=0.0, size_before_mb=0.0, size_after_mb=0.0, error="VACUUM 失敗"
        )

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "❌ DB メンテナンス失敗")
        self.assertEqual(_fields_map(embed)["❌ エラー"], "VACUUM 失敗")


class TestSendBackupCompletion(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_success_sends_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_backup_completion(
            backup_path="/backups/db_2026.duckdb",
            size_mb=1234.5,
            elapsed_seconds=5.0,
            pruned_count=3,
        )

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "✅ DB バックアップ完了")
        fields = _fields_map(embed)
        self.assertEqual(fields["⏱ 処理時間"], "5.0 秒")
        self.assertEqual(fields["💾 サイズ"], "1,234.50 MB")
        self.assertEqual(fields["🗑 削除世代"], "3 件")
        self.assertEqual(fields["📁 保存先"], "/backups/db_2026.duckdb")

    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_error_sends_error_field(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_backup_completion(
            backup_path="", size_mb=0.0, elapsed_seconds=0.0, pruned_count=0, error="ディスク不足"
        )

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "❌ DB バックアップ失敗")
        self.assertEqual(_fields_map(embed)["❌ エラー"], "ディスク不足")


class TestSendMonthlyReportNotification(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_sends_metrics_as_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_monthly_report_notification(
            target_month="2026-05",
            net_return=0.1234,
            max_drawdown=-0.05,
            sharpe_ratio=1.5,
            hit_rate=0.55,
            avg_slippage=0.001,
            symbol_count=1234,
            report_path="/reports/2026-05.md",
        )

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "📊 月次レポート生成完了")
        fields = _fields_map(embed)
        self.assertEqual(fields["📅 対象月"], "2026-05")
        self.assertEqual(fields["💹 Net Return"], "12.34%")
        self.assertEqual(fields["📈 Sharpe Ratio"], "1.50")
        self.assertEqual(fields["🏷 集計銘柄数"], "1,234")
        self.assertEqual(fields["📁 保存先"], "/reports/2026-05.md")

    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_none_metrics_render_na(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_monthly_report_notification(
            target_month="2026-05",
            net_return=None,
            max_drawdown=None,
            sharpe_ratio=None,
            hit_rate=None,
            avg_slippage=None,
            symbol_count=None,
        )

        self.assertTrue(result)
        fields = _fields_map(_embed_from_post(mock_post))
        self.assertEqual(fields["💹 Net Return"], "N/A")
        self.assertEqual(fields["🏷 集計銘柄数"], "N/A")
        self.assertNotIn("📁 保存先", fields)


class TestSendHitRateDriftAlert(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_drifted_sends_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))
        result_obj = SimpleNamespace(
            is_drifted=True,
            current_week="2026-W22",
            current_hit_rate=0.40,
            avg_hit_rate=0.52,
            drop_ratio=0.12,
            alert_threshold=0.10,
            alert_weeks=4,
        )

        result = send_hit_rate_drift_alert(result_obj)

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "⚠️ モデルドリフト検知: Hit Rate 低下")
        fields = _fields_map(embed)
        self.assertEqual(fields["📅 週"], "2026-W22")
        self.assertEqual(fields["🎯 当週 Hit Rate"], "40.0%")
        self.assertEqual(fields["📊 過去 4 週平均"], "52.0%")
        self.assertEqual(fields["📉 低下率"], "12.0%")
        self.assertEqual(fields["🚧 閾値"], "10.0%")

    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_not_drifted_does_not_send(self, mock_post):
        result_obj = SimpleNamespace(
            is_drifted=False,
            current_week="2026-W22",
            current_hit_rate=0.55,
            avg_hit_rate=0.55,
            drop_ratio=0.0,
            alert_threshold=0.10,
            alert_weeks=4,
        )

        result = send_hit_rate_drift_alert(result_obj)

        self.assertFalse(result)
        mock_post.assert_not_called()


class TestSendShadowEvaluationNotification(unittest.TestCase):
    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_challenger_wins_sends_fields(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_shadow_evaluation_notification(
            {
                "challenger_wins": True,
                "production_hit_rate": 0.50,
                "production_sharpe": 1.0,
                "n_production": 1000,
                "challenger_hit_rate": 0.55,
                "challenger_sharpe": 1.2,
                "n_challenger": 1500,
            }
        )

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "🏆 A/Bテスト: Challenger 昇格候補")
        self.assertIn("promote_challenger_to_production", embed["description"])
        fields = _fields_map(embed)
        self.assertIn("Hit Rate: 0.500", fields["🏭 Production"])
        self.assertIn("n=1,000", fields["🏭 Production"])
        self.assertIn("Hit Rate: 0.550", fields["🧪 Challenger"])
        self.assertIn("n=1,500", fields["🧪 Challenger"])

    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_no_winner_has_no_description(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        result = send_shadow_evaluation_notification({"challenger_wins": False})

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "ℹ️ A/Bテスト: 評価完了")
        self.assertEqual(embed["description"], "")
        fields = _fields_map(embed)
        self.assertIn("n=0", fields["🏭 Production"])


class TestSendWebhookNotification(unittest.TestCase):
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    @patch(
        "src.reporting.discord.webhook_sender.isoformat_jst",
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

    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_fields_are_attached_to_embed(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))
        fields = [{"name": "🕐 時刻", "value": "09:00", "inline": True}]

        result = send_webhook_notification("fields-title", "", fields=fields)

        self.assertTrue(result)
        payload = mock_post.call_args.kwargs["json_payload"]
        self.assertEqual(payload["embeds"][0]["fields"], fields)

    @patch(
        "src.reporting.discord.webhook_sender._rate_limiter.check_and_record",
        return_value=(True, None),
    )
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    def test_no_fields_key_when_not_provided(self, mock_post, _mock_rl):
        mock_post.return_value = MagicMock(raise_for_status=MagicMock(return_value=None))

        send_webhook_notification("no-fields-title", "message")

        payload = mock_post.call_args.kwargs["json_payload"]
        self.assertNotIn("fields", payload["embeds"][0])


class TestChunkedTextHelpers(unittest.TestCase):
    def test_split_text_chunks_preserves_lines_when_possible(self):
        chunks = split_text_chunks("a\nb\nc", limit=3, preserve_lines=True)

        self.assertEqual(chunks, ["a\nb", "c"])

    @patch("src.reporting.discord.webhook_sender.send_webhook_text", return_value=True)
    def test_send_webhook_text_chunked_sends_all_chunks(self, mock_send):
        result = send_webhook_text_chunked("12345", limit=2, preserve_lines=False)

        self.assertTrue(result)
        self.assertEqual([call.args[0] for call in mock_send.call_args_list], ["12", "34", "5"])


class TestSendShapNotification(unittest.TestCase):
    @patch("src.reporting.discord.webhook_sender.send_webhook_text", return_value=True)
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

    @patch("src.reporting.discord.webhook_sender.send_webhook_text", return_value=True)
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
    @patch("src.reporting.discord.webhook_sender.send_webhook_text", return_value=True)
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

    @patch("src.reporting.discord.webhook_sender._get_webhook_url")
    def test_returns_none_when_no_webhook_url(self, mock_url):
        """Webhook URL 未設定時は None が返ること"""
        from src.reporting.discord.discord_utils import _post_webhook

        mock_url.return_value = ""
        result = _post_webhook(json_payload={"content": "test"})
        assert result is None

    @patch("src.reporting.discord.webhook_sender._get_webhook_url")
    @patch("src.reporting.discord.webhook_sender.requests.post")
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

    @patch("src.reporting.discord.webhook_sender.send_webhook_text")
    def test_sends_single_chunk_for_short_text(self, mock_send):
        """短いテキストは1チャンクで送信されること"""
        from src.reporting.discord.discord_utils import send_webhook_text_chunked

        mock_send.return_value = True
        result = send_webhook_text_chunked("Hello", limit=2000)
        assert result is True
        mock_send.assert_called_once()

    @patch("src.reporting.discord.webhook_sender.send_webhook_text")
    def test_returns_false_if_any_chunk_fails(self, mock_send):
        """一部チャンクが失敗した場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_webhook_text_chunked

        # send_webhook_text が False を返すとき全体も False になること
        mock_send.return_value = False
        result = send_webhook_text_chunked("Hello\nWorld\nTest", limit=2000)
        assert result is False


class TestSendTextFileChunked:
    """send_text_file_chunked のテスト"""

    @patch("src.reporting.discord.webhook_sender.send_webhook_text_chunked")
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

    @patch("src.reporting.discord.webhook_sender.send_webhook_notification")
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
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    @patch("src.reporting.discord.webhook_sender.isoformat_jst")
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
    @patch("src.reporting.discord.webhook_sender._post_webhook")
    @patch("src.reporting.discord.webhook_sender.isoformat_jst")
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
        "src.reporting.discord.webhook_sender._get_webhook_url",
        return_value="https://webhook.url/test",
    )
    @patch("src.reporting.discord.webhook_sender.requests.post")
    def test_posts_json_payload(self, mock_post, mock_url):
        """有効な URL がある場合に requests.post が呼ばれること"""
        from src.reporting.discord.discord_utils import _post_webhook

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        _post_webhook(json_payload={"content": "test"})
        mock_post.assert_called_once()

    @patch("src.reporting.discord.webhook_sender._get_webhook_url", return_value=None)
    def test_returns_none_when_no_url(self, mock_url):
        """URL 未設定時は None が返ること"""
        from src.reporting.discord.discord_utils import _post_webhook

        result = _post_webhook(json_payload={"content": "test"})
        self.assertIsNone(result)


class TestSendWebhookFile(unittest.TestCase):
    """send_webhook_file のテスト"""

    @patch("src.reporting.discord.webhook_sender._post_webhook")
    @patch("builtins.open", mock_open(read_data=b"test data"))
    @patch("src.reporting.discord.webhook_sender.os.path.exists", return_value=True)
    def test_sends_file_when_exists(self, mock_exists, mock_post):
        """ファイルが存在する場合に _post_webhook が呼ばれ True が返ること"""
        from src.reporting.discord.discord_utils import send_webhook_file

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = send_webhook_file("/tmp/test.csv", "test comment")
        mock_post.assert_called_once()
        self.assertTrue(result)

    @patch("src.reporting.discord.webhook_sender.os.path.exists", return_value=False)
    def test_skips_when_file_not_exists(self, mock_exists):
        """ファイルが存在しない場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_webhook_file

        result = send_webhook_file("/tmp/nonexistent.csv", "test")
        self.assertFalse(result)

    @patch("src.reporting.discord.webhook_sender._post_webhook", return_value=None)
    @patch("builtins.open", mock_open(read_data=b"test data"))
    @patch("src.reporting.discord.webhook_sender.os.path.exists", return_value=True)
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

    @patch("src.reporting.discord.webhook_sender.send_webhook_text", return_value=True)
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

    @patch("src.prediction.db.load_drift_summary")
    def test_returns_false_when_accuracy_df_none_and_db_empty(self, mock_load):
        """accuracy_df=None かつ DB が空の場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_weekly_report

        mock_load.return_value = pd.DataFrame()
        result = send_weekly_report(accuracy_df=None)
        self.assertFalse(result)


class TestSendFeatureSuggestionNotification(unittest.TestCase):
    @patch("src.reporting.discord.discord_utils.send_webhook_notification", return_value=True)
    def test_sends_notification_with_candidates(self, mock_send):
        from src.reporting.discord.discord_utils import send_feature_suggestion_notification

        candidates = pd.DataFrame(
            [
                {"feature": "rsi", "importance_mean": 0.001, "importance_rank": 50},
                {"feature": "macd", "importance_mean": 0.0005, "importance_rank": 49},
            ]
        )
        result = send_feature_suggestion_notification(
            [{"market": "jp", "symbol": "7203", "candidates": candidates}]
        )
        self.assertTrue(result)
        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        self.assertIn("特徴量除外提案", kwargs["title"])
        self.assertIn("rsi", kwargs["message"])
        self.assertIn("macd", kwargs["message"])

    @patch("src.reporting.discord.discord_utils.send_webhook_notification", return_value=True)
    def test_global_warning_for_common_features(self, mock_send):
        from src.reporting.discord.discord_utils import send_feature_suggestion_notification

        candidates_df = pd.DataFrame(
            [{"feature": "rsi", "importance_mean": 0.001, "importance_rank": 50}]
        )
        result = send_feature_suggestion_notification(
            [
                {"market": "jp", "symbol": "7203", "candidates": candidates_df},
                {"market": "jp", "symbol": "7201", "candidates": candidates_df.copy()},
            ],
            global_threshold=2,
        )
        self.assertTrue(result)
        message = mock_send.call_args.kwargs["message"]
        self.assertIn("グローバル除外候補", message)
        self.assertIn("rsi", message)

    @patch("src.reporting.discord.discord_utils.send_webhook_notification")
    def test_returns_false_for_empty_list(self, mock_send):
        from src.reporting.discord.discord_utils import send_feature_suggestion_notification

        result = send_feature_suggestion_notification([])
        self.assertFalse(result)
        mock_send.assert_not_called()

    @patch("src.reporting.discord.discord_utils.send_webhook_notification", return_value=True)
    def test_handles_empty_candidates_df(self, mock_send):
        from src.reporting.discord.discord_utils import send_feature_suggestion_notification

        result = send_feature_suggestion_notification(
            [{"market": "us", "symbol": "AAPL", "candidates": pd.DataFrame()}]
        )
        self.assertTrue(result)
        message = mock_send.call_args.kwargs["message"]
        self.assertIn("除外候補なし", message)


class TestSendAccuracySummary(unittest.TestCase):
    """send_accuracy_summary のテスト"""

    @patch("src.reporting.discord.discord_utils.send_webhook_text", return_value=True)
    def test_sends_summary_with_valid_df(self, mock_send):
        """有効な summary_df がある場合に通知が送信されること"""
        from src.reporting.discord.discord_utils import send_accuracy_summary

        summary_df = pd.DataFrame(
            {
                "market": ["jp", "us"],
                "symbol": ["7203", "AAPL"],
                "direction_accuracy": [0.6, 0.75],
                "mean_abs_error": [0.01, 0.02],
                "n_samples": [30, 20],
            }
        )
        result = send_accuracy_summary(summary_df, horizon=1)
        mock_send.assert_called_once()
        self.assertTrue(result)
        text = mock_send.call_args[0][0]
        self.assertIn("予測精度サマリー", text)
        self.assertIn("jp/7203", text)
        self.assertIn("us/AAPL", text)

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    def test_returns_false_for_empty_df(self, mock_send):
        """空 DataFrame の場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_accuracy_summary

        result = send_accuracy_summary(pd.DataFrame())
        mock_send.assert_not_called()
        self.assertFalse(result)

    @patch("src.reporting.discord.discord_utils.send_webhook_text")
    def test_returns_false_for_none(self, mock_send):
        """None を渡した場合は False が返ること"""
        from src.reporting.discord.discord_utils import send_accuracy_summary

        result = send_accuracy_summary(None)
        mock_send.assert_not_called()
        self.assertFalse(result)
