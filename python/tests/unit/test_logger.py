"""src.utils.logger モジュールのユニットテスト"""

import contextlib
import logging
import os
import sys
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


class TestMaskingFormatter(unittest.TestCase):
    """_MaskingFormatter の機密値マスキングテスト"""

    def _make_formatter(self, env_overrides: dict):
        """env_overrides を適用した _MaskingFormatter を返す"""
        from src.utils.logger import _DATE_FORMAT, _LOG_FORMAT, _MaskingFormatter

        with patch.dict(os.environ, env_overrides, clear=False):
            return _MaskingFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    def _format_msg(self, formatter, msg: str) -> str:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return formatter.format(record)

    def test_webhook_url_is_masked(self):
        """DISCORD_WEBHOOK_URL の値がログメッセージ中でマスクされること"""
        secret = "https://discord.com/api/webhooks/123456/ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        formatter = self._make_formatter({"DISCORD_WEBHOOK_URL": secret})
        result = self._format_msg(formatter, f"Error: {secret}")
        self.assertNotIn(secret, result)
        self.assertIn("***REDACTED***", result)

    def test_token_is_masked(self):
        """DISCORD_BOT_TOKEN の値がマスクされること"""
        token = "MTIzNDU2Nzg5.ABCDEF.xyz-ABCDEFGHIJKLMNOPQ"
        formatter = self._make_formatter({"DISCORD_BOT_TOKEN": token})
        result = self._format_msg(formatter, f"token={token}")
        self.assertNotIn(token, result)
        self.assertIn("***REDACTED***", result)

    def test_short_value_not_masked(self):
        """8文字未満の値はマスク対象外であること"""
        formatter = self._make_formatter({"DISCORD_WEBHOOK_URL": "short"})
        result = self._format_msg(formatter, "short value here")
        # "short" はマスクされない
        self.assertIn("short", result)

    def test_non_sensitive_env_not_masked(self):
        """機密キーワードを含まない環境変数の値はマスクされないこと"""
        formatter = self._make_formatter({"MY_REGULAR_VAR": "regularvalue123"})
        result = self._format_msg(formatter, "regularvalue123 present")
        self.assertIn("regularvalue123", result)

    def test_exception_traceback_is_masked(self):
        """例外トレースバック内の機密値もマスクされること"""
        secret = "https://discord.com/api/webhooks/99999/SECRETTOKEN_ABCDEFGHIJKLMNOP"
        formatter = self._make_formatter({"DISCORD_WEBHOOK_URL": secret})

        try:
            raise ValueError(f"HTTPError for url: {secret}")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="request failed",
            args=(),
            exc_info=exc_info,
        )
        result = formatter.format(record)
        self.assertNotIn(secret, result)
        self.assertIn("***REDACTED***", result)

    def test_no_sensitive_env_no_pattern(self):
        """機密環境変数が存在しない場合は _pattern が None でエラーなしにフォーマットされること"""
        # 既存の機密変数を一時的に除去して確認
        env_without_sensitive = {
            k: v
            for k, v in os.environ.items()
            if not any(
                kw in k.upper()
                for kw in (
                    "DISCORD_",
                    "TOKEN",
                    "SECRET",
                    "PASSWORD",
                    "WEBHOOK",
                    "API_KEY",
                    "AUTH",
                    "CREDENTIAL",
                )
            )
        }
        from src.utils.logger import _DATE_FORMAT, _LOG_FORMAT, _MaskingFormatter

        with patch.dict(os.environ, env_without_sensitive, clear=True):
            formatter = _MaskingFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        self.assertIsNone(formatter._pattern)
        result = self._format_msg(formatter, "normal message")
        self.assertIn("normal message", result)


if __name__ == "__main__":
    unittest.main()
