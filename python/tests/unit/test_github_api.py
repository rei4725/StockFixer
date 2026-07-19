from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.utils.github_api import get_issue, list_recently_merged_pull_requests


class TestListRecentlyMergedPullRequests:
    @patch("src.utils.github_api.requests.get")
    def test_returns_only_merged_prs(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "number": 101,
                "body": "Closes #564",
                "merge_commit_sha": "abc123",
                "merged_at": "2026-07-19T01:00:00Z",
                "updated_at": "2026-07-19T01:00:00Z",
            },
            {
                "number": 102,
                "body": "WIP",
                "merge_commit_sha": None,
                "merged_at": None,
                "updated_at": "2026-07-19T01:00:00Z",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = list_recently_merged_pull_requests(since=datetime(2026, 7, 1))

        assert len(result) == 1
        assert result[0] == {"number": 101, "body": "Closes #564", "merge_commit_sha": "abc123"}

    @patch("src.utils.github_api.requests.get")
    def test_filters_by_since(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "number": 201,
                "body": "Closes #1",
                "merge_commit_sha": "old",
                "merged_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = list_recently_merged_pull_requests(since=datetime(2026, 7, 1))

        assert result == []

    @patch("src.utils.github_api.requests.get")
    def test_raises_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 401")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="HTTP 401"):
            list_recently_merged_pull_requests(since=datetime(2026, 7, 1))


class TestGetIssue:
    @patch("src.utils.github_api.requests.get")
    def test_returns_number_title_labels(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "number": 564,
            "title": "[factory:fb44f0011174] AND合成ルール (jp)",
            "labels": [{"name": "strategy-factory"}, {"name": "auto-ok"}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_issue(564)

        assert result == {
            "number": 564,
            "title": "[factory:fb44f0011174] AND合成ルール (jp)",
            "labels": ["strategy-factory", "auto-ok"],
        }
