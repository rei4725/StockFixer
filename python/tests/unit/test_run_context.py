"""src.utils.run_context のユニットテスト"""

import threading
import unittest


class TestRunContext(unittest.TestCase):
    def setUp(self):
        from src.utils.run_context import _current

        _current.set(None)

    def test_get_run_id_returns_none_initially(self):
        from src.utils.run_context import get_run_id

        self.assertIsNone(get_run_id())

    def test_new_run_context_sets_run_id(self):
        from src.utils.run_context import get_run_id, new_run_context

        ctx = new_run_context()
        self.assertIsNotNone(get_run_id())
        self.assertEqual(get_run_id(), ctx.run_id)

    def test_run_id_is_12_char_hex(self):
        from src.utils.run_context import new_run_context

        ctx = new_run_context()
        self.assertEqual(len(ctx.run_id), 12)
        int(ctx.run_id, 16)  # 16進数として解釈できること

    def test_set_run_context(self):
        from src.utils.run_context import RunContext, get_run_id, set_run_context

        ctx = RunContext(run_id="abc123def456")
        set_run_context(ctx)
        self.assertEqual(get_run_id(), "abc123def456")

    def test_each_new_run_context_has_unique_id(self):
        from src.utils.run_context import new_run_context

        ctx1 = new_run_context()
        ctx2 = new_run_context()
        self.assertNotEqual(ctx1.run_id, ctx2.run_id)

    def test_thread_inherits_no_context(self):
        """別スレッドはメインスレッドの run_id を継承しない"""
        from src.utils.run_context import _current, get_run_id, new_run_context

        new_run_context()
        captured = []

        def worker():
            _current.set(None)
            captured.append(get_run_id())

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertIsNone(captured[0])

    def test_masking_formatter_includes_run_id(self):
        """_MaskingFormatter が run_id をログに含めること"""
        import logging
        from unittest.mock import patch

        from src.utils.logger import _DATE_FORMAT, _LOG_FORMAT, _MaskingFormatter
        from src.utils.run_context import new_run_context

        ctx = new_run_context()
        formatter = _MaskingFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        self.assertIn(ctx.run_id, result)

    def test_masking_formatter_uses_dash_when_no_context(self):
        """run_id 未設定時は '-' が出力されること"""
        import logging

        from src.utils.logger import _DATE_FORMAT, _LOG_FORMAT, _MaskingFormatter

        formatter = _MaskingFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        self.assertIn("[-]", result)


if __name__ == "__main__":
    unittest.main()
