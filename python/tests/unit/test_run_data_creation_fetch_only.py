"""run_data_creation.run_single の --fetch-only の道（#1391）。"""

import unittest
from unittest.mock import patch

import pandas as pd
import run_data_creation as cli

from src.market_data.quality_check import QualityCheckResult, QualityIssue


class TestRunSingleFetchOnly(unittest.TestCase):
    def test_five_returns_with_data_logs_row_count(self):
        data = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        quality = QualityCheckResult(market="us", symbol="AAPL", passed=True)
        result = ("us", "AAPL", data, data, quality)
        with (
            patch.object(cli, "fetch_stock_data_with_features", return_value=result),
            patch.object(cli, "logger") as logger,
        ):
            cli.run_single("us", "AAPL", fetch_only=True)
        logger.error.assert_not_called()
        logger.info.assert_called_once()
        self.assertIn("3行", logger.info.call_args[0][0])

    def test_data_none_warns_with_quality_issues(self):
        raw = pd.DataFrame({"close": [1.0, 2.0]})
        quality = QualityCheckResult(
            market="jp",
            symbol="7203",
            passed=False,
            issues=[QualityIssue(check="split_gap", level="error", detail="段差あり")],
        )
        result = ("jp", "7203", None, raw, quality)
        with (
            patch.object(cli, "fetch_stock_data_with_features", return_value=result),
            patch.object(cli, "logger") as logger,
        ):
            cli.run_single("jp", "7203", fetch_only=True)
        logger.error.assert_not_called()
        logger.info.assert_not_called()
        logger.warning.assert_called_once()
        message = logger.warning.call_args[0][0]
        self.assertIn("jp/7203", message)
        self.assertIn("split_gap", message)
        self.assertIn("段差あり", message)

    def test_data_none_without_quality_result_warns(self):
        raw = pd.DataFrame({"close": [1.0]})
        result = ("us", "MSFT", None, raw, None)
        with (
            patch.object(cli, "fetch_stock_data_with_features", return_value=result),
            patch.object(cli, "logger") as logger,
        ):
            cli.run_single("us", "MSFT", fetch_only=True)
        logger.error.assert_not_called()
        logger.warning.assert_called_once()
        self.assertIn("us/MSFT", logger.warning.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
