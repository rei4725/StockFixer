"""ユニットテスト: watchlist/manager.py の追加カバレッジ"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


def _make_diff(market: str, added: list, removed: list):
    from src.watchlist.manager import WatchlistDiff

    diff = WatchlistDiff(market=market)
    diff.added = added
    diff.removed = removed
    return diff


class TestLoadSaveWatchlist(unittest.TestCase):
    def test_load_watchlist_reads_json(self):
        from src.watchlist.manager import _load_watchlist

        data = {"jp": ["7203", "6758"], "us": ["AAPL"]}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", encoding="utf-8", delete=False
        ) as f:
            json.dump(data, f)
            fname = f.name
        try:
            with patch("src.watchlist.manager.get_watchlist_path", return_value=fname):
                result = _load_watchlist()
        finally:
            os.unlink(fname)
        self.assertEqual(result["jp"], ["7203", "6758"])

    def test_save_watchlist_writes_json(self):
        from src.watchlist.manager import _save_watchlist

        data = {"jp": ["7203", "6758"]}
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", encoding="utf-8", delete=False
        ) as f:
            json.dump({}, f)
            fname = f.name
        try:
            with patch("src.watchlist.manager.get_watchlist_path", return_value=fname):
                _save_watchlist(data)
            with open(fname, encoding="utf-8") as f:
                result = json.load(f)
        finally:
            os.unlink(fname)
        self.assertEqual(result["jp"], ["7203", "6758"])


class TestAppendAuditLog(unittest.TestCase):
    def test_creates_log_when_not_exists(self):
        from src.watchlist.manager import WatchlistDiff, _append_audit_log

        diff = WatchlistDiff(market="jp", added=["9999"], removed=["7777"])

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "watchlist_audit_log.json")
            with patch("src.watchlist.manager.get_watchlist_audit_log_path", return_value=log_path):
                _append_audit_log([diff])
            with open(log_path, encoding="utf-8") as f:
                log = json.load(f)

        self.assertEqual(len(log), 1)
        self.assertIn("jp", log[0]["markets"])
        self.assertIn("9999", log[0]["markets"]["jp"]["added"])

    def test_appends_to_existing_log(self):
        from src.watchlist.manager import WatchlistDiff, _append_audit_log

        diff = WatchlistDiff(market="us", added=["AAPL"])

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "watchlist_audit_log.json")
            # 既存ログ作成
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump([{"existing": True}], f)
            with patch("src.watchlist.manager.get_watchlist_audit_log_path", return_value=log_path):
                _append_audit_log([diff])
            with open(log_path, encoding="utf-8") as f:
                log = json.load(f)

        self.assertEqual(len(log), 2)
        self.assertTrue(log[0]["existing"])
        self.assertIn("us", log[1]["markets"])


class TestApplyWatchlistUpdate(unittest.TestCase):
    def test_applies_additions_and_removals(self):
        from src.watchlist.manager import WatchlistDiff, apply_watchlist_update

        diff = WatchlistDiff(market="jp", added=["9999"], removed=["1111"])

        initial_data = {"jp": ["1111", "7203"]}

        def mock_load():
            return initial_data.copy()

        saved_data = {}

        def mock_save(data):
            saved_data.update(data)

        with (
            patch("src.watchlist.manager._load_watchlist", side_effect=mock_load),
            patch("src.watchlist.manager._save_watchlist", side_effect=mock_save),
            patch("src.watchlist.manager._append_audit_log"),
        ):
            apply_watchlist_update([diff])

        jp_list = saved_data.get("jp", [])
        self.assertIn("9999", jp_list)
        self.assertNotIn("1111", jp_list)

    def test_skips_diff_without_changes(self):
        from src.watchlist.manager import WatchlistDiff, apply_watchlist_update

        diff = WatchlistDiff(market="jp")

        with (
            patch("src.watchlist.manager._load_watchlist", return_value={}),
            patch("src.watchlist.manager._save_watchlist") as mock_save,
            patch("src.watchlist.manager._append_audit_log"),
        ):
            apply_watchlist_update([diff])

        mock_save.assert_called_once()


class TestRunWatchlistRefresh(unittest.TestCase):
    def test_returns_empty_diffs_when_fetch_fails(self):
        from src.watchlist.manager import run_watchlist_refresh

        with (
            patch("src.watchlist.manager._load_watchlist", return_value={"jp_stock": ["7203"]}),
            patch("src.watchlist.manager.fetch_index_symbols", return_value=[]),
        ):
            result = run_watchlist_refresh(markets=["jp"])

        self.assertEqual(result, [])

    def test_returns_diffs_when_symbols_fetched(self):
        from src.watchlist.manager import run_watchlist_refresh

        with (
            patch("src.watchlist.manager._load_watchlist", return_value={"jp_stock": ["7203"]}),
            patch("src.watchlist.manager.fetch_index_symbols", return_value=["7203", "6758"]),
            patch("src.watchlist.manager.save_index_membership_snapshot"),
            patch("src.watchlist.manager.diff_watchlist") as mock_diff,
            patch("src.watchlist.manager.apply_watchlist_update"),
        ):
            mock_result = MagicMock()
            mock_result.added = ["6758"]
            mock_result.removed = []
            mock_result.has_changes = True
            mock_diff.return_value = mock_result
            result = run_watchlist_refresh(markets=["jp"])

        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
