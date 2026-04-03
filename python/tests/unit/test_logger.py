"""src.utils.logger モジュールのユニットテスト"""

import contextlib
import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestGetLogger(unittest.TestCase):
    """get_logger のテスト"""

    def test_returns_logger(self):
        """get_logger が Logger インスタンスを返すこと"""
        from src.utils.logger import get_logger

        logger = get_logger("test.module")
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_name_matches(self):
        """返ったロガーの名前が引数と一致すること"""
        from src.utils.logger import get_logger

        logger = get_logger("my.custom.module")
        self.assertEqual(logger.name, "my.custom.module")

    def test_same_name_returns_same_logger(self):
        """同じ名前で get_logger を2回呼ぶと同一のオブジェクトが返ること"""
        from src.utils.logger import get_logger

        logger1 = get_logger("duplicate.module")
        logger2 = get_logger("duplicate.module")
        self.assertIs(logger1, logger2)


class TestConfigureRoot(unittest.TestCase):
    """_configure_root のブランチカバレッジテスト（ファイルハンドラ・Streamをモック）"""

    def _apply_handler_patches(self, stack, tmp_dir):
        """ExitStack にハンドラ関連パッチを積む"""
        from logging.handlers import RotatingFileHandler

        stack.enter_context(
            patch(
                "src.utils.logger.RotatingFileHandler",
                return_value=MagicMock(spec=RotatingFileHandler),
            )
        )
        stack.enter_context(
            patch(
                "src.utils.logger.logging.StreamHandler",
                return_value=MagicMock(spec=logging.StreamHandler),
            )
        )
        # io.TextIOWrapper が sys.stderr.buffer を閉じないようにモックする
        stack.enter_context(patch("src.utils.logger.io.TextIOWrapper", return_value=MagicMock()))
        stack.enter_context(patch("src.utils.logger._LOG_DIR", tmp_dir))
        stack.enter_context(patch("os.makedirs"))

    def _patch_handlers(self, tmp_dir):
        """RotatingFileHandler と StreamHandler をモックするコンテキストのリストを返す"""
        from logging.handlers import RotatingFileHandler

        return [
            patch(
                "src.utils.logger.RotatingFileHandler",
                return_value=MagicMock(spec=RotatingFileHandler),
            ),
            patch(
                "src.utils.logger.logging.StreamHandler",
                return_value=MagicMock(spec=logging.StreamHandler),
            ),
            patch("src.utils.logger._LOG_DIR", tmp_dir),
            patch("os.makedirs"),
        ]

    def test_configure_with_existing_handlers_sets_configured(self):
        """ルートロガーにハンドラが既存の場合 early return して _root_configured が True になること"""
        import src.utils.logger as logger_module

        root_logger = logging.getLogger()
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)
        try:
            logger_module._root_configured = False
            logger_module._configure_root()
            self.assertTrue(logger_module._root_configured)
        finally:
            root_logger.removeHandler(dummy_handler)
            logger_module._root_configured = True

    def test_log_level_debug_from_env(self):
        """LOG_LEVEL=DEBUG のとき DEBUG レベルで構成されること"""
        import src.utils.logger as logger_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}))
                self._apply_handler_patches(stack, tmp_dir)

                logger_module._root_configured = False
                root_logger = logging.getLogger()
                original_handlers = root_logger.handlers[:]
                root_logger.handlers.clear()
                try:
                    logger_module._configure_root()
                    self.assertTrue(logger_module._root_configured)
                    self.assertEqual(root_logger.level, logging.DEBUG)
                finally:
                    root_logger.handlers[:] = original_handlers
                    logger_module._root_configured = True

    def test_configure_twice_is_noop(self):
        """_configure_root を2回呼んでも2回目は no-op になること"""
        import src.utils.logger as logger_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            patches = self._patch_handlers(tmp_dir)
            with patches[2], patches[3]:  # _LOG_DIR と makedirs だけモック
                logger_module._root_configured = True  # 1回目済みの状態
                logger_module._configure_root()  # 2回目はno-op
                # _root_configured が変わらず True のまま
                self.assertTrue(logger_module._root_configured)

    def test_invalid_log_level_falls_back_to_info(self):
        """不正な LOG_LEVEL は INFO にフォールバックすること"""
        import src.utils.logger as logger_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {"LOG_LEVEL": "NOTAVALIDLEVEL"}))
                self._apply_handler_patches(stack, tmp_dir)

                logger_module._root_configured = False
                root_logger = logging.getLogger()
                original_handlers = root_logger.handlers[:]
                root_logger.handlers.clear()
                try:
                    logger_module._configure_root()
                    # getattr(logging, "NOTAVALIDLEVEL", logging.INFO) → INFO
                    self.assertEqual(root_logger.level, logging.INFO)
                finally:
                    root_logger.handlers[:] = original_handlers
                    logger_module._root_configured = True


if __name__ == "__main__":
    unittest.main()
