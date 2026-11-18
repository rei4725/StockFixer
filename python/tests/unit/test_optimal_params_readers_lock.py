"""置き場を決めた後に自分で開く読み手が、書き手と同じロックの内で読むこと（#2007）。

置き場を決めた直後に書き手がロックを取った形を、置き場の決めを差し替えて作る。
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock


class _ReaderLockBase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        results_dir = Path(self._dir.name) / "results"
        results_dir.mkdir()
        self.results_path = results_dir / "optimal_params.json"
        self.results_path.write_text(json.dumps({"jp_7203": {"threshold": 0.01}}), encoding="utf-8")

    def tearDown(self):
        self._dir.cleanup()

    def _run_reader_while_writer_holds_lock(self, read, broken: str | None) -> dict:
        """書き手がロックを持って置き換えている最中に読み手を走らせ、結果を返す。

        broken が None なら古いファイルが無い一瞬を、文字列なら書きかけの中身を模す。
        """
        out: dict = {}

        def target() -> None:
            try:
                out["value"] = read()
            except Exception as e:  # noqa: BLE001 - 読み手の例外を本体のスレッドで見る
                out["error"] = e

        with FileLock(f"{self.results_path}.lock"):
            if broken is None:
                self.results_path.unlink()
            else:
                self.results_path.write_text(broken, encoding="utf-8")
            reader = threading.Thread(target=target)
            reader.start()
            time.sleep(0.3)
            self.results_path.write_text(
                json.dumps({"jp_7203": {"threshold": 0.02}}), encoding="utf-8"
            )
        reader.join(timeout=10)
        self.assertFalse(reader.is_alive())
        return out


class TestPersistenceReaderLock(_ReaderLockBase):
    """batch の飛ばしの判定が読む persistence.get_optimal_params"""

    def _read(self):
        from src.backtest.optimizer.persistence import get_optimal_params

        return get_optimal_params("jp", "7203")

    def test_waits_for_writer_when_file_vanished(self):
        with patch(
            "src.utils.optimal_params_loader.optimal_params_read_path",
            return_value=self.results_path,
        ):
            out = self._run_reader_while_writer_holds_lock(self._read, broken=None)
        self.assertEqual(out.get("value"), {"threshold": 0.02})

    def test_waits_for_writer_when_file_half_written(self):
        with patch(
            "src.utils.optimal_params_loader.optimal_params_read_path",
            return_value=self.results_path,
        ):
            out = self._run_reader_while_writer_holds_lock(self._read, broken="{")
        self.assertNotIn("error", out)
        self.assertEqual(out.get("value"), {"threshold": 0.02})


class TestPredictionPipelineReaderLock(_ReaderLockBase):
    """日次の予想が読む prediction_pipeline.get_optimal_params"""

    def _read(self):
        from src.prediction.prediction_pipeline import get_optimal_params

        return get_optimal_params("jp", "7203")

    def test_waits_for_writer_when_file_vanished(self):
        with patch(
            "src.prediction.prediction_pipeline.optimal_params_read_path",
            return_value=self.results_path,
        ):
            out = self._run_reader_while_writer_holds_lock(self._read, broken=None)
        self.assertEqual(out.get("value"), {"threshold": 0.02})

    def test_waits_for_writer_when_file_half_written(self):
        with patch(
            "src.prediction.prediction_pipeline.optimal_params_read_path",
            return_value=self.results_path,
        ):
            out = self._run_reader_while_writer_holds_lock(self._read, broken="{")
        self.assertEqual(out.get("value"), {"threshold": 0.02})

    def test_reads_without_lock_when_wait_times_out(self):
        """ロックを待ちきれないときは、今の読み手と同じくロックの外で読む（止まらない）"""
        with (
            patch("src.utils.optimal_params_loader._LOCK_TIMEOUT_SEC", 0.2),
            patch(
                "src.prediction.prediction_pipeline.optimal_params_read_path",
                return_value=self.results_path,
            ),
        ):
            out: dict = {}
            with FileLock(f"{self.results_path}.lock"):
                reader = threading.Thread(target=lambda: out.update(value=self._read()))
                reader.start()
                reader.join(timeout=10)
        self.assertEqual(out.get("value"), {"threshold": 0.01})


if __name__ == "__main__":
    unittest.main()
