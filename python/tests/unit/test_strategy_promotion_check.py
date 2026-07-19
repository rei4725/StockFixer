from unittest.mock import patch

from src.orchestration.jobs.periodic import run_strategy_promotion_check


class TestRunStrategyPromotionCheck:
    @patch("config.settings.STRATEGY_PROMOTION_CHECK_ENABLED", False)
    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_skips_when_disabled_and_not_forced(self, mock_list):
        run_strategy_promotion_check(force=False)
        mock_list.assert_not_called()

    @patch("src.reporting.discord.discord_utils.send_strategy_promotion_detected")
    @patch("src.utils.db.strategy_promotions.save_strategy_promotion")
    @patch("src.utils.db.strategy_promotions.promotion_exists", return_value=False)
    @patch("src.backtest.promotion_detection.load_gate_baseline", return_value=1.25)
    @patch("src.utils.github_api.get_issue")
    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_detects_and_records_factory_pr(
        self, mock_list, mock_get_issue, mock_baseline, mock_exists, mock_save, mock_notify
    ):
        mock_list.return_value = [
            {"number": 564, "body": "Closes #999", "merge_commit_sha": "abc123"}
        ]
        mock_get_issue.return_value = {
            "number": 999,
            "title": "[factory:fb44f0011174] AND合成ルール (jp)",
            "labels": ["strategy-factory"],
        }

        run_strategy_promotion_check(force=True)

        mock_save.assert_called_once_with(
            pr_number=564,
            merge_commit_hash="abc123",
            rule_or_feature_id="fb44f0011174",
            pre_promotion_baseline=1.25,
        )
        mock_notify.assert_called_once_with(
            pr_number=564, rule_or_feature_id="fb44f0011174", pre_promotion_baseline=1.25
        )

    @patch("src.utils.db.strategy_promotions.save_strategy_promotion")
    @patch("src.utils.db.strategy_promotions.promotion_exists", return_value=False)
    @patch("src.utils.github_api.get_issue")
    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_ignores_pr_not_linked_to_factory_issue(
        self, mock_list, mock_get_issue, mock_exists, mock_save
    ):
        mock_list.return_value = [{"number": 1, "body": "Closes #2", "merge_commit_sha": "x"}]
        mock_get_issue.return_value = {"number": 2, "title": "普通のバグ修正", "labels": ["bug"]}

        run_strategy_promotion_check(force=True)

        mock_save.assert_not_called()

    @patch("src.utils.db.strategy_promotions.save_strategy_promotion")
    @patch("src.utils.db.strategy_promotions.promotion_exists", return_value=True)
    @patch("src.utils.github_api.get_issue")
    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_skips_already_recorded_pr(self, mock_list, mock_get_issue, mock_exists, mock_save):
        mock_list.return_value = [
            {"number": 564, "body": "Closes #999", "merge_commit_sha": "abc123"}
        ]

        run_strategy_promotion_check(force=True)

        mock_get_issue.assert_not_called()
        mock_save.assert_not_called()

    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_does_not_raise_on_github_api_failure(self, mock_list):
        mock_list.side_effect = Exception("network error")
        run_strategy_promotion_check(force=True)  # 例外を送出しないことを確認
