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
        held = patch("src.utils.db.stock_features.load_open_position_symbols", return_value=set())
        self.held = held.start()
        self.addCleanup(held.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def test_keeps_held_and_short_symbols_outside_watchlist(self):
        """watchlist に無い保有・空売りの銘柄が対象に入ること（#1615）"""
        from src.utils.db.stock_features import get_active_symbols

        self.held.return_value = {"AVB", "9984"}
        wl = self._watchlist(self.tmpdir, {"us": ["AAPL"], "jp": ["7203"]})
        all_symbols = [("jp", "7203"), ("jp", "9984"), ("us", "AAPL"), ("us", "AVB")]
        with (
            patch("src.utils.db.stock_features.get_all_symbols", return_value=all_symbols),
            patch("src.utils.db.stock_features.get_watchlist_path", return_value=wl),
        ):
            self.assertEqual(get_active_symbols(), all_symbols)

    def test_unheld_symbol_outside_watchlist_stays_excluded(self):
        """保有していない外れた銘柄は今までどおり対象に入らないこと（#717, #1615）"""
        from src.utils.db.stock_features import get_active_symbols

        self.held.return_value = {"AVB"}
        wl = self._watchlist(self.tmpdir, {"us": ["AAPL"], "jp": []})
        with (
            patch(
                "src.utils.db.stock_features.get_all_symbols",
                return_value=[("us", "AAPL"), ("us", "AVB"), ("us", "OLDCO")],
            ),
            patch("src.utils.db.stock_features.get_watchlist_path", return_value=wl),
        ):
            self.assertEqual(get_active_symbols(), [("us", "AAPL"), ("us", "AVB")])

    def test_held_symbol_without_features_is_not_invented(self):
        """特徴量の無い保有は返さないこと（予想を出せず、取得の対象の側で足す）"""
        from src.utils.db.stock_features import get_active_symbols

        self.held.return_value = {"AVB"}
        wl = self._watchlist(self.tmpdir, {"us": ["AAPL"], "jp": []})
        with (
            patch("src.utils.db.stock_features.get_all_symbols", return_value=[("us", "AAPL")]),
            patch("src.utils.db.stock_features.get_watchlist_path", return_value=wl),
        ):
            self.assertEqual(get_active_symbols(), [("us", "AAPL")])

    def test_continues_with_watchlist_when_holdings_unreadable(self):
        """保有を読めない日は例外を上へ出さず ERROR を出して watchlist で続けること（#1615）"""
        from src.utils.db.stock_features import get_active_symbols

        self.held.side_effect = RuntimeError("db down")
        wl = self._watchlist(self.tmpdir, {"us": ["AAPL"], "jp": []})
        with (
            patch(
                "src.utils.db.stock_features.get_all_symbols",
                return_value=[("us", "AAPL"), ("us", "AVB")],
            ),
            patch("src.utils.db.stock_features.get_watchlist_path", return_value=wl),
            self.assertLogs("src.utils.db.stock_features", level="ERROR"),
        ):
            self.assertEqual(get_active_symbols(), [("us", "AAPL")])

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
