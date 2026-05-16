"""src.prediction.manager のアーティファクト保存・ロード機能テスト"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd

from src.prediction.manager import ModelManager, _compute_feature_hash, _get_git_sha
from src.prediction.models.base import BaseModel


class _MockModel(BaseModel):
    """テスト用のシンプルなモデル実装"""

    def __init__(self, model_name: str = "MockModel"):
        super().__init__(model_name)
        self.model = {"weights": [1.0, 2.0]}

    def train(self, X: pd.DataFrame, y: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series([0.0] * len(X))

    def save_model(self, path: str):
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        self.model = joblib.load(path)


class TestComputeFeatureHash(unittest.TestCase):
    def test_same_columns_produce_same_hash(self):
        cols = ["a", "b", "c"]
        self.assertEqual(_compute_feature_hash(cols), _compute_feature_hash(cols))

    def test_different_order_produces_different_hash(self):
        self.assertNotEqual(
            _compute_feature_hash(["a", "b"]),
            _compute_feature_hash(["b", "a"]),
        )

    def test_different_columns_produce_different_hash(self):
        self.assertNotEqual(
            _compute_feature_hash(["x"]),
            _compute_feature_hash(["y"]),
        )

    def test_returns_hex_string(self):
        result = _compute_feature_hash(["col1"])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 32)


class TestGetGitSha(unittest.TestCase):
    def test_returns_string(self):
        sha = _get_git_sha()
        self.assertIsInstance(sha, str)

    def test_fallback_to_empty_on_failure(self):
        with patch("src.prediction.manager.subprocess.run", side_effect=OSError("no git")):
            self.assertEqual(_get_git_sha(), "")


class TestModelManagerSaveArtifact(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(model_dir=self.tmp_dir)
        self.manager._registered_models = {"MockModel": _MockModel}
        self.manager.create_model("MockModel", "test_model")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_creates_dict_artifact(self):
        """save_model が dict 形式で保存すること"""
        self.manager.save_model("test_model", feature_columns=["f1", "f2"])
        path = os.path.join(self.tmp_dir, "test_model.joblib")
        artifact = joblib.load(path)
        self.assertIsInstance(artifact, dict)
        self.assertIn("model", artifact)
        self.assertIn("feature_hash", artifact)
        self.assertIn("git_sha", artifact)
        self.assertIn("trained_at", artifact)

    def test_save_records_feature_hash(self):
        """feature_columns から hash が計算されること"""
        cols = ["f1", "f2", "f3"]
        self.manager.save_model("test_model", feature_columns=cols)
        path = os.path.join(self.tmp_dir, "test_model.joblib")
        artifact = joblib.load(path)
        self.assertEqual(artifact["feature_hash"], _compute_feature_hash(cols))

    def test_save_without_feature_columns_sets_hash_none(self):
        """feature_columns 未指定時は feature_hash が None"""
        self.manager.save_model("test_model")
        path = os.path.join(self.tmp_dir, "test_model.joblib")
        artifact = joblib.load(path)
        self.assertIsNone(artifact["feature_hash"])


class TestModelManagerLoadArtifact(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(model_dir=self.tmp_dir)
        self.manager._registered_models = {"MockModel": _MockModel}

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _save_dict_artifact(self, model_name, feature_hash, raw_model=None):
        if raw_model is None:
            raw_model = {"weights": [1.0]}
        artifact = {
            "model": raw_model,
            "feature_hash": feature_hash,
            "git_sha": "abc1234",
            "trained_at": "2026-01-01T00:00:00+00:00",
        }
        path = os.path.join(self.tmp_dir, f"{model_name}.joblib")
        joblib.dump(artifact, path)

    def _save_legacy_artifact(self, model_name, raw_model=None):
        if raw_model is None:
            raw_model = {"weights": [1.0]}
        path = os.path.join(self.tmp_dir, f"{model_name}.joblib")
        joblib.dump(raw_model, path)

    def test_load_new_format_sets_model(self):
        """新形式 artifact から model が正しくロードされること"""
        raw = {"weights": [3.0]}
        self._save_dict_artifact("my_model", feature_hash="abc123", raw_model=raw)
        self.manager.create_model("MockModel", "my_model")
        loaded = self.manager.load_model("my_model")
        self.assertEqual(loaded.model, raw)

    def test_load_legacy_format_sets_model(self):
        """旧形式（rawオブジェクト）も後方互換でロードできること"""
        raw = {"weights": [5.0]}
        self._save_legacy_artifact("old_model", raw_model=raw)
        self.manager.create_model("MockModel", "old_model")
        loaded = self.manager.load_model("old_model")
        self.assertEqual(loaded.model, raw)

    def test_load_logs_warning_on_hash_mismatch(self):
        """feature_hash 不一致時に logger.warning が呼ばれること"""
        cols_at_train = ["f1", "f2"]
        self._save_dict_artifact(
            "mismatch_model", feature_hash=_compute_feature_hash(cols_at_train)
        )
        self.manager.create_model("MockModel", "mismatch_model")

        cols_at_predict = ["f1", "f2", "f3_new"]
        with self.assertLogs("src.prediction.manager", level="WARNING") as cm:
            self.manager.load_model("mismatch_model", feature_columns=cols_at_predict)
        self.assertTrue(any("特徴量不一致" in line for line in cm.output))

    def test_load_no_warning_on_hash_match(self):
        """feature_hash が一致する場合は warning が出ないこと"""
        cols = ["f1", "f2"]
        self._save_dict_artifact("match_model", feature_hash=_compute_feature_hash(cols))
        self.manager.create_model("MockModel", "match_model")

        with self.assertLogs("src.prediction.manager", level="DEBUG") as cm:
            self.manager.load_model("match_model", feature_columns=cols)
        warning_lines = [line for line in cm.output if "WARNING" in line and "特徴量不一致" in line]
        self.assertEqual(len(warning_lines), 0)

    def test_load_no_warning_when_feature_columns_not_given(self):
        """feature_columns が None のときは比較をスキップすること"""
        cols = ["f1", "f2"]
        self._save_dict_artifact("skip_check_model", feature_hash=_compute_feature_hash(cols))
        self.manager.create_model("MockModel", "skip_check_model")
        # feature_columns を指定しないので warning は出ない（ログ未検出でも失敗しない）
        self.manager.load_model("skip_check_model")

    def test_load_sends_discord_on_mismatch(self):
        """feature_hash 不一致時に Discord 通知が試みられること"""
        cols_train = ["x1"]
        self._save_dict_artifact("discord_model", feature_hash=_compute_feature_hash(cols_train))
        self.manager.create_model("MockModel", "discord_model")

        with patch("src.reporting.discord.discord_utils.send_webhook_notification") as mock_notify:
            with self.assertLogs("src.prediction.manager", level="WARNING") as cm:
                self.manager.load_model("discord_model", feature_columns=["x1", "x2_new"])

        self.assertTrue(any("特徴量不一致" in line for line in cm.output))
        mock_notify.assert_called_once()


class TestTrainModelPassesFeatureColumns(unittest.TestCase):
    """train_model が feature_columns を save_model に渡すこと"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(model_dir=self.tmp_dir)
        self.manager._registered_models = {"MockModel": _MockModel}
        self.manager.create_model("MockModel", "train_model")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_train_model_saves_feature_hash(self):
        """train_model 後に保存された artifact に feature_hash が入ること"""
        X = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        y = pd.Series([0.1, 0.2])
        self.manager.train_model("train_model", X, y)

        path = os.path.join(self.tmp_dir, "train_model.joblib")
        artifact = joblib.load(path)
        self.assertEqual(artifact["feature_hash"], _compute_feature_hash(["col_a", "col_b"]))


if __name__ == "__main__":
    unittest.main()
