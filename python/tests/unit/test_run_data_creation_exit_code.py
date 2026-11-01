"""run_data_creation の終了コード（#1470）。"""

import sys
import unittest
from unittest.mock import patch

import run_data_creation as cli

from src.domain.types import BatchFailure, BatchResult


def _run_main(argv: list) -> int:
    with (
        patch("src.orchestration.port_wiring.wire_ports"),
        patch.object(sys, "argv", ["run_data_creation.py", *argv]),
    ):
        try:
            cli.main()
        except SystemExit as e:
            return int(e.code or 0)
    return 0


class TestSingleExitCode(unittest.TestCase):
    def test_save_exception_exits_one_with_error_log(self):
        with (
            patch.object(cli, "save_stock_data_with_features", side_effect=RuntimeError("db")),
            patch.object(cli, "logger") as logger,
        ):
            code = _run_main(["--market", "us", "--symbol", "AAPL"])
        self.assertEqual(code, 1)
        logger.error.assert_called()

    def test_save_success_exits_zero(self):
        with (
            patch.object(cli, "save_stock_data_with_features", return_value=None),
            patch.object(cli, "logger") as logger,
        ):
            code = _run_main(["--market", "us", "--symbol", "AAPL"])
        self.assertEqual(code, 0)
        logger.error.assert_not_called()

    def test_fetch_only_skip_exits_zero(self):
        with (
            patch.object(cli, "fetch_stock_data_with_features", return_value=None),
            patch.object(cli, "logger") as logger,
        ):
            code = _run_main(["--market", "us", "--symbol", "AAPL", "--fetch-only"])
        self.assertEqual(code, 0)
        logger.error.assert_not_called()

    def test_fetch_only_exception_exits_one(self):
        with (
            patch.object(cli, "fetch_stock_data_with_features", side_effect=RuntimeError("net")),
            patch.object(cli, "logger") as logger,
        ):
            code = _run_main(["--market", "us", "--symbol", "AAPL", "--fetch-only"])
        self.assertEqual(code, 1)
        logger.error.assert_called()


class TestBatchExitCode(unittest.TestCase):
    def _run_batch(self, result: BatchResult, extra: list | None = None):
        with (
            patch.object(cli, "load_target_symbols", return_value=[]),
            patch.object(cli, "run_batch_pipeline", return_value=result) as pipeline,
            patch.object(cli, "logger") as logger,
        ):
            code = _run_main(["--batch", *(extra or [])])
        return code, logger, pipeline

    def test_failed_exits_one_with_count(self):
        failure = BatchFailure(market="us", symbol="AAPL", error="save")
        code, logger, _ = self._run_batch(BatchResult(failed=[failure]))
        self.assertEqual(code, 1)
        logger.error.assert_called_once()
        self.assertIn("1", str(logger.error.call_args))

    def test_empty_failed_exits_zero(self):
        code, logger, _ = self._run_batch(BatchResult(succeeded=["ok"], skipped=["s"]))
        self.assertEqual(code, 0)
        logger.error.assert_not_called()

    def test_fetch_only_failed_exits_one(self):
        failure = BatchFailure(market="jp", symbol="7203", error="timeout")
        code, _, pipeline = self._run_batch(BatchResult(failed=[failure]), ["--fetch-only"])
        self.assertEqual(code, 1)
        self.assertTrue(pipeline.call_args.kwargs["fetch_only"])


if __name__ == "__main__":
    unittest.main()
