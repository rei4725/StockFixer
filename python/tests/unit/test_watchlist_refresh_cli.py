"""週ごとの銘柄リストの更新の CLI（run_watchlist_refresh.py）と Step Summary のテスト（#808）

ワークフローの yml に直書きした Python は、テストも lint も通らないため、import の道が
壊れても（fe88ee03 で src.services が消えた）約半年気づけなかった。CLI に移し、
取得の失敗（fetch_failed）を exit 1 で見えるようにする。
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.watchlist.manager import WatchlistDiff, failed_markets, format_refresh_summary

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "weekly-watchlist-refresh.yml"


class TestFailedMarkets:
    def test_returns_markets_with_fetch_failed(self):
        diffs = [
            WatchlistDiff(market="us", fetch_failed=True),
            WatchlistDiff(market="jp", added=["7203"]),
        ]
        assert failed_markets(diffs) == ["us"]

    def test_empty_when_all_fetched(self):
        assert failed_markets([WatchlistDiff(market="us"), WatchlistDiff(market="jp")]) == []


class TestFormatRefreshSummary:
    def test_no_changes(self):
        text = format_refresh_summary([WatchlistDiff(market="us"), WatchlistDiff(market="jp")])
        assert "変更なし" in text

    def test_fetch_failed_is_not_reported_as_no_changes(self):
        """取得に失敗した市場を「変更なし」と書かない"""
        text = format_refresh_summary(
            [WatchlistDiff(market="us", fetch_failed=True), WatchlistDiff(market="jp")]
        )
        assert "変更なし" not in text
        assert "US" in text
        assert "取得失敗" in text

    def test_lists_added_removed_and_unverified(self):
        diff = WatchlistDiff(
            market="jp", added=["7203"], removed=["9984"], removed_unverified=["6758"]
        )
        text = format_refresh_summary([diff])
        assert "### JP" in text
        assert "**追加** (1): 7203" in text
        assert "**削除** (1): 9984" in text
        assert "**保留** (1): 6758" in text

    def test_truncates_long_lists(self):
        diff = WatchlistDiff(market="us", added=[f"S{i}" for i in range(25)])
        text = format_refresh_summary([diff])
        assert "S19" in text
        assert "S20" not in text
        assert "…他" in text


class TestCliMain:
    @patch("src.watchlist.manager.run_watchlist_refresh")
    def test_exit_1_and_writes_summary_when_fetch_failed(self, mock_refresh, tmp_path):
        from run_watchlist_refresh import main

        mock_refresh.return_value = [
            WatchlistDiff(market="us", fetch_failed=True),
            WatchlistDiff(market="jp"),
        ]
        summary = tmp_path / "summary.md"

        assert main(["--summary-path", str(summary)]) == 1
        assert "取得失敗" in summary.read_text(encoding="utf-8")

    @patch("src.watchlist.manager.run_watchlist_refresh")
    def test_exit_0_when_all_fetched(self, mock_refresh, tmp_path):
        from run_watchlist_refresh import main

        mock_refresh.return_value = [WatchlistDiff(market="us"), WatchlistDiff(market="jp")]
        summary = tmp_path / "summary.md"

        assert main(["--summary-path", str(summary)]) == 0
        assert "変更なし" in summary.read_text(encoding="utf-8")

    @patch("src.watchlist.manager.run_watchlist_refresh")
    def test_summary_path_defaults_to_github_step_summary(
        self, mock_refresh, tmp_path, monkeypatch
    ):
        from run_watchlist_refresh import main

        mock_refresh.return_value = [WatchlistDiff(market="us")]
        summary = tmp_path / "gh_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

        assert main([]) == 0
        assert summary.exists()


class TestRunNowWatchlist:
    @patch("src.orchestration.scheduler.run_weekly_watchlist_refresh")
    def test_run_now_watchlist_invokes_refresh(self, mock_run):
        """週の更新は GitHub Actions へ移したため SCHEDULE_CONFIG に無い。手動の道は直接呼ぶ"""
        from run_scheduler import run_now

        run_now("watchlist")
        mock_run.assert_called_once()


class TestWorkflowUsesCli:
    def test_workflow_calls_cli_instead_of_inline_python(self):
        """yml に Python を直書きしない（import の道が壊れても CI で気づけないため）"""
        if not _WORKFLOW.exists():
            pytest.skip("リポジトリの外で動いている")
        text = _WORKFLOW.read_text(encoding="utf-8")
        assert "python run_watchlist_refresh.py" in text
        assert "from src." not in text
