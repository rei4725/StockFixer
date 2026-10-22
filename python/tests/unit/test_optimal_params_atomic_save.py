"""save_optimal_params_json の保存の安全性（Issue 888）。

ロックの内で読み直す・一時ファイルと os.replace で置き換える・
壊れた JSON は退避して ERROR を出し保存を止める、を確かめる。
"""

import glob
import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.optimizer import persistence


def _result_df() -> pd.DataFrame:
    return pd.DataFrame({"threshold": [0.02], "sharpe_ratio": [1.5], "num_trades": [50]})


class _TmpConfigDirMixin(unittest.TestCase):
    """persistence の保存先 config/ を一時フォルダへ向ける。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        # dirname を4回たどると root になる位置に __file__ を置く
        fake_file = os.path.join(root, "src", "backtest", "optimizer", "persistence.py")
        file_patch = patch.object(persistence, "__file__", fake_file)
        file_patch.start()
        self.addCleanup(file_patch.stop)
        self.addCleanup(self._tmp.cleanup)
        self.config_dir = os.path.join(root, "config")
        os.makedirs(self.config_dir, exist_ok=True)
        self.json_path = os.path.join(self.config_dir, "optimal_params.json")

    def _read(self) -> dict:
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)


class TestCorruptedJsonIsQuarantined(_TmpConfigDirMixin):
    def test_corrupted_json_is_moved_aside_and_not_overwritten(self):
        broken = '{"jp_1301": {"threshold": 0.01}, "jp_13'
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write(broken)

        with patch.object(persistence.logger, "error") as mock_error:
            with self.assertRaises(persistence.OptimalParamsCorruptedError):
                persistence.save_optimal_params_json(_result_df(), "jp", "7203")

        quarantined = glob.glob(f"{self.json_path}.corrupt-*")
        self.assertEqual(len(quarantined), 1)
        with open(quarantined[0], "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), broken)
        mock_error.assert_called()
        # 1銘柄だけのファイルで上書きしない
        self.assertFalse(os.path.exists(self.json_path))

    def test_non_dict_json_is_quarantined(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)

        with patch.object(persistence.logger, "error"):
            with self.assertRaises(persistence.OptimalParamsCorruptedError):
                persistence.save_optimal_params_json(_result_df(), "jp", "7203")

        self.assertEqual(len(glob.glob(f"{self.json_path}.corrupt-*")), 1)
        self.assertFalse(os.path.exists(self.json_path))


class TestConcurrentWritersKeepBothSymbols(_TmpConfigDirMixin):
    def test_two_writers_keep_both_symbols(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"jp_1301": {"threshold": 0.01}}, f)

        real_load = json.load
        other_writer_errors: list = []

        def other_writer() -> None:
            try:
                persistence.save_optimal_params_json(_result_df(), "jp", "6758")
            except Exception as e:  # pragma: no cover - 失敗はテストで検出する
                other_writer_errors.append(e)

        other = threading.Thread(target=other_writer)
        state = {"started": False}

        def load_then_let_other_writer_run(fp, *args, **kwargs):
            data = real_load(fp, *args, **kwargs)
            if not state["started"] and threading.current_thread() is main_thread:
                # 1つ目の書き手が読んだ直後に、2つ目の書き手を走らせる
                state["started"] = True
                other.start()
                other.join(timeout=1.0)
            return data

        main_thread = threading.current_thread()
        with patch.object(persistence.json, "load", side_effect=load_then_let_other_writer_run):
            persistence.save_optimal_params_json(_result_df(), "jp", "7203")
            other.join(timeout=30)

        self.assertFalse(other.is_alive())
        self.assertEqual(other_writer_errors, [])
        saved = self._read()
        self.assertIn("jp_1301", saved)
        self.assertIn("jp_7203", saved)
        self.assertIn("jp_6758", saved)


class TestAtomicReplace(_TmpConfigDirMixin):
    def test_write_goes_through_temp_file_and_os_replace(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"jp_1301": {"threshold": 0.01}}, f)

        with patch.object(persistence.os, "replace", wraps=os.replace) as mock_replace:
            persistence.save_optimal_params_json(_result_df(), "jp", "7203")

        mock_replace.assert_called_once()
        src, dst = mock_replace.call_args[0]
        self.assertEqual(os.path.abspath(dst), os.path.abspath(self.json_path))
        self.assertEqual(os.path.dirname(os.path.abspath(src)), os.path.abspath(self.config_dir))
        self.assertNotEqual(os.path.abspath(src), os.path.abspath(self.json_path))
        saved = self._read()
        self.assertIn("jp_1301", saved)
        self.assertIn("jp_7203", saved)

    def test_failure_while_writing_keeps_original_file(self):
        original = {"jp_1301": {"threshold": 0.01}}
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(original, f)

        with patch.object(persistence.json, "dump", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                persistence.save_optimal_params_json(_result_df(), "jp", "7203")

        self.assertEqual(self._read(), original)
        leftovers = [
            name
            for name in os.listdir(self.config_dir)
            if name not in ("optimal_params.json", "optimal_params.json.lock")
        ]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
