"""
ユニットテスト: optimal_params_loader

tmp_path を使ったファイルベースの単体テスト。
外部ファイルシステム以外の依存はなし。
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch


class TestGetOptimalParams(unittest.TestCase):
    """get_optimal_params() のファイルベーステスト"""

    def _get_fn(self):
        from src.utils.optimal_params_loader import get_optimal_params

        return get_optimal_params

    def test_returns_params_when_file_and_key_exist(self, tmp_path=None):
        """ファイルが存在し、対象キーがある場合はパラメータ辞書を返す"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "optimal_params.json"
            data = {"jp_7203": {"threshold": 0.01, "stop_loss_pct": 0.03, "take_profit_pct": 0.07}}
            path.write_text(json.dumps(data), encoding="utf-8")

            get_optimal_params = self._get_fn()
            with patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", path):
                result = get_optimal_params("jp", "7203")

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["threshold"], 0.01)
        self.assertAlmostEqual(result["stop_loss_pct"], 0.03)
        self.assertAlmostEqual(result["take_profit_pct"], 0.07)

    def test_returns_none_when_file_not_exists(self):
        """ファイルが存在しない場合は None を返す"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            missing_path = Path(d) / "does_not_exist.json"

            get_optimal_params = self._get_fn()
            with patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", missing_path):
                result = get_optimal_params("jp", "7203")

        self.assertIsNone(result)

    def test_returns_none_when_key_not_in_json(self):
        """ファイルは存在するが対象キーがない場合は None を返す"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "optimal_params.json"
            data = {"us_AAPL": {"threshold": 0.02}}
            path.write_text(json.dumps(data), encoding="utf-8")

            get_optimal_params = self._get_fn()
            with patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", path):
                result = get_optimal_params("jp", "9999")

        self.assertIsNone(result)

    def _assert_unreadable(self, content: str) -> None:
        import tempfile

        from src.utils.optimal_params_loader import OptimalParamsUnreadableError

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "optimal_params.json"
            path.write_text(content, encoding="utf-8")

            get_optimal_params = self._get_fn()
            with patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", path):
                with self.assertLogs("src.utils.optimal_params_loader", level="ERROR"):
                    with self.assertRaises(OptimalParamsUnreadableError):
                        get_optimal_params("jp", "7203")

    def test_raises_when_json_is_invalid(self):
        """ファイルが在るのに不正な JSON の場合は、未登録の None にせず例外を投げ ERROR を出す"""
        self._assert_unreadable("{ invalid json !!!}")

    def test_raises_when_top_level_is_not_dict(self):
        """最上位が dict でない場合は、例外を投げ ERROR を出す"""
        self._assert_unreadable(json.dumps([{"jp_7203": {"threshold": 0.01}}]))

    def test_raises_when_file_cannot_be_read(self):
        """ファイルが在るのに開けない場合は、例外を投げ ERROR を出す"""
        import tempfile

        from src.utils.optimal_params_loader import OptimalParamsUnreadableError

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "optimal_params.json"
            path.write_text("{}", encoding="utf-8")

            get_optimal_params = self._get_fn()
            with patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", path):
                with patch.object(Path, "open", side_effect=PermissionError("denied")):
                    with self.assertLogs("src.utils.optimal_params_loader", level="ERROR"):
                        with self.assertRaises(OptimalParamsUnreadableError):
                            get_optimal_params("jp", "7203")


class TestReadDuringWeeklySave(unittest.TestCase):
    """週の最適化の置き換えの一瞬に、読み手が無いと見て落ちないこと（#1996）。"""

    def setUp(self):
        import tempfile

        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        self.results_dir = root / "results"
        self.results_dir.mkdir()
        self.results_path = self.results_dir / "optimal_params.json"
        self.seed_path = root / "config" / "optimal_params.json"
        self.seed_path.parent.mkdir()
        self.seed_path.write_text(json.dumps({"jp_7203": {"threshold": 0.99}}), encoding="utf-8")
        self.results_path.write_text(json.dumps({"jp_7203": {"threshold": 0.01}}), encoding="utf-8")
        self._patches = [
            patch.dict("os.environ", {"STOCKFIXER_RESULTS_DIR": str(self.results_dir)}),
            patch("src.utils.optimal_params_loader.OPTIMAL_PARAMS_PATH", self.seed_path),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._dir.cleanup()

    def test_waits_for_writer_lock_and_returns_new_content(self):
        """書き手がロックを持って置き換えている最中は、待ってから新しい中身を返す"""
        import threading
        import time

        from filelock import FileLock

        from src.utils.optimal_params_loader import get_optimal_params

        results: dict = {}
        with FileLock(f"{self.results_path}.lock"):
            # 置き換えの一瞬（古いファイルは無く、新しいファイルはまだ無い）を模す
            self.results_path.unlink()
            reader = threading.Thread(
                target=lambda: results.update(value=get_optimal_params("jp", "7203"))
            )
            reader.start()
            time.sleep(0.3)
            self.results_path.write_text(
                json.dumps({"jp_7203": {"threshold": 0.02}}), encoding="utf-8"
            )
        reader.join(timeout=10)

        self.assertFalse(reader.is_alive())
        self.assertEqual(results["value"], {"threshold": 0.02})

    def test_does_not_fall_back_to_seed_when_results_vanish_for_an_instant(self):
        """一度在った results/ が一瞬無く見えても、config/ の初めの値を返さない"""
        from src.utils.optimal_params_loader import get_optimal_params

        self.assertEqual(get_optimal_params("jp", "7203"), {"threshold": 0.01})

        real_exists = Path.exists
        calls = {"n": 0}

        def flaky_exists(path_self, *args, **kwargs):
            if path_self == self.results_path and calls["n"] == 0:
                calls["n"] += 1
                return False
            return real_exists(path_self, *args, **kwargs)

        with patch.object(Path, "exists", flaky_exists):
            result = get_optimal_params("jp", "7203")

        self.assertEqual(calls["n"], 1)
        self.assertEqual(result, {"threshold": 0.01})

    def test_raises_instead_of_seed_when_results_stay_missing(self):
        """一度在った results/ が待っても無いままなら、初めの値にせず例外を投げる"""
        from src.utils.optimal_params_loader import (
            OptimalParamsUnreadableError,
            get_optimal_params,
        )

        self.assertEqual(get_optimal_params("jp", "7203"), {"threshold": 0.01})
        self.results_path.unlink()

        with patch("src.utils.optimal_params_loader._VANISHED_RETRY_WAIT_SEC", 0.0):
            with self.assertLogs("src.utils.optimal_params_loader", level="ERROR"):
                with self.assertRaises(OptimalParamsUnreadableError):
                    get_optimal_params("jp", "7203")

    def test_raises_when_lock_wait_times_out(self):
        """ロックを待ちきれないときは、例外を投げる（既定の閾値で黙って進めない）"""
        import threading

        from filelock import FileLock

        from src.utils.optimal_params_loader import (
            OptimalParamsUnreadableError,
            get_optimal_params,
        )

        errors: dict = {}

        def read() -> None:
            try:
                get_optimal_params("jp", "7203")
            except OptimalParamsUnreadableError as e:
                errors["e"] = e

        with patch("src.utils.optimal_params_loader._LOCK_TIMEOUT_SEC", 0.2):
            with FileLock(f"{self.results_path}.lock"):
                reader = threading.Thread(target=read)
                reader.start()
                reader.join(timeout=10)

        self.assertIn("e", errors)


if __name__ == "__main__":
    unittest.main()
