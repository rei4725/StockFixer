"""Unit tests: 予測対象を watchlist.json に従わせる

stock_features には上場廃止銘柄の履歴が残るため、予測対象を DB の
全銘柄から取ると廃止銘柄が Top10 に混ざる（#717 後に発覚）。
"""

import json
import unittest
from unittest.mock import patch


class TestGetActiveSymbols(unittest.TestCase):
    """get_active_symbols のフィルタ挙動"""

    def _watchlist(self, tmpdir, data):
        path = tmpdir / "watchlist.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def setUp(self):
        import tempfile
        from pathlib import Path

        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_excludes_symbols_not_in_watchlist(self):
        """watchlist にない銘柄（上場廃止など）が除外されること"""
        from src.utils.db.stock_features import get_active_symbols

        wl = self._watchlist(self.tmpdir, {"us": ["AAPL", "VMRK"], "jp": ["7203"]})
        with (
            patch(
                "src.utils.db.stock_features.get_all_symbols",
                return_value=[("us", "AAPL"), ("us", "AVB"), ("us", "VMRK"), ("jp", "7203")],
            ),
            patch("src.utils.db.stock_features.get_watchlist_path", return_value=wl),
        ):
            # 入力順（DB 側で market, symbol ソート済み）を保つこと
            self.assertEqual(get_active_symbols(), [("us", "AAPL"), ("us", "VMRK"), ("jp", "7203")])

    def test_does_not_invent_symbols_without_features(self):
        """watchlist にあっても特徴量が無い銘柄は返さないこと"""
        from src.utils.db.stock_features import get_active_symbols

        wl = self._watchlist(self.tmpdir, {"us": ["AAPL", "NEWCO"], "jp": []})
        with (
            patch(
                "src.utils.db.stock_features.get_all_symbols",
                return_value=[("us", "AAPL")],
            ),
            patch("src.utils.db.stock_features.get_watchlist_path", return_value=wl),
        ):
            self.assertEqual(get_active_symbols(), [("us", "AAPL")])

    def test_case_insensitive_match(self):
        """大文字小文字が違っても一致すること"""
        from src.utils.db.stock_features import get_active_symbols

        wl = self._watchlist(self.tmpdir, {"us": ["aapl"], "jp": []})
        with (
            patch(
                "src.utils.db.stock_features.get_all_symbols",
                return_value=[("us", "AAPL")],
            ),
            patch("src.utils.db.stock_features.get_watchlist_path", return_value=wl),
        ):
            self.assertEqual(get_active_symbols(), [("us", "AAPL")])

    def test_falls_back_to_all_when_watchlist_unreadable(self):
        """watchlist が読めないときは全銘柄を返す（予測が丸ごと止まるのを防ぐ）"""
        from src.utils.db.stock_features import get_active_symbols

        with (
            patch(
                "src.utils.db.stock_features.get_all_symbols",
                return_value=[("us", "AAPL"), ("us", "AVB")],
            ),
            patch(
                "src.utils.db.stock_features.get_watchlist_path",
                return_value=str(self.tmpdir / "missing.json"),
            ),
        ):
            self.assertEqual(get_active_symbols(), [("us", "AAPL"), ("us", "AVB")])

    def test_falls_back_to_all_when_watchlist_empty(self):
        """watchlist が空でも全銘柄を返す（安全側に倒す）"""
        from src.utils.db.stock_features import get_active_symbols

        wl = self._watchlist(self.tmpdir, {"us": [], "jp": []})
        with (
            patch(
                "src.utils.db.stock_features.get_all_symbols",
                return_value=[("us", "AAPL")],
            ),
            patch("src.utils.db.stock_features.get_watchlist_path", return_value=wl),
        ):
            self.assertEqual(get_active_symbols(), [("us", "AAPL")])


class TestPredictionUsesActiveSymbols(unittest.TestCase):
    """予測系が get_active_symbols を使っていること"""

    def test_prediction_modules_do_not_call_get_all_symbols(self):
        """予測系モジュールが get_all_symbols を直接呼んでいないこと"""
        import pathlib

        targets = [
            "src/prediction/prediction_pipeline.py",
            "src/prediction/predict_unified.py",
            "src/prediction/shadow_evaluation.py",
        ]
        offenders = []
        for t in targets:
            text = pathlib.Path(t).read_text(encoding="utf-8")
            if "get_all_symbols()" in text:
                offenders.append(t)
        self.assertEqual(offenders, [], f"get_all_symbols() が残っている: {offenders}")


if __name__ == "__main__":
    unittest.main()
