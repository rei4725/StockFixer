"""run_longterm_backtest.py CLI の引数パース。"""

import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

import run_longterm_backtest as cli


class TestExecutionLagArgument(unittest.TestCase):
    def test_negative_execution_lag_is_rejected_with_clear_message(self):
        argv = ["run_longterm_backtest.py", "--execution-lag", "-1"]
        stderr = StringIO()
        with patch.object(sys, "argv", argv), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                cli.parse_args()
        # argparse は使用法エラーを終了コード 2 で返す。
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("execution_lag", stderr.getvalue())

    def test_default_execution_lag_is_one(self):
        argv = ["run_longterm_backtest.py"]
        with patch.object(sys, "argv", argv):
            args = cli.parse_args()
        self.assertEqual(args.execution_lag, 1)

    def test_explicit_zero_is_accepted(self):
        argv = ["run_longterm_backtest.py", "--execution-lag", "0"]
        with patch.object(sys, "argv", argv):
            args = cli.parse_args()
        self.assertEqual(args.execution_lag, 0)


if __name__ == "__main__":
    unittest.main()
