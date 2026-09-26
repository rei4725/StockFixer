"""ユニットテスト: PostgresPredictionResultRepository（prediction BC 所有の SQL への委譲）。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.domain.ports import PredictionResultRepository
from src.infrastructure.persistence.prediction_result_repository import (
    PostgresPredictionResultRepository,
)

_MOD = "src.prediction.db.prediction_results"


class TestPostgresPredictionResultRepository(unittest.TestCase):
    def test_implements_port(self):
        self.assertIsInstance(PostgresPredictionResultRepository(), PredictionResultRepository)

    def test_latest_timestamp_delegates(self):
        with patch(f"{_MOD}.load_latest_prediction_timestamp", return_value="ts") as mock_load:
            result = PostgresPredictionResultRepository().latest_timestamp(
                model_version="challenger"
            )
        mock_load.assert_called_once_with(model_version="challenger")
        self.assertEqual(result, "ts")

    def test_results_at_delegates(self):
        sentinel = pd.DataFrame({"x": [1]})
        with patch(f"{_MOD}.load_prediction_results", return_value=sentinel) as mock_load:
            result = PostgresPredictionResultRepository().results_at(
                predicted_at="ts", market="jp", top_n=10
            )
        mock_load.assert_called_once_with(predicted_at="ts", market="jp", top_n=10, worst_n=None)
        self.assertIs(result, sentinel)

    def test_markets_at_delegates(self):
        with patch(f"{_MOD}.load_prediction_markets", return_value=["jp"]) as mock_load:
            result = PostgresPredictionResultRepository().markets_at("ts")
        mock_load.assert_called_once_with(predicted_at="ts")
        self.assertEqual(result, ["jp"])

    def test_get_latest_by_market_delegates(self):
        sentinel = pd.DataFrame({"x": [1]})
        with patch(f"{_MOD}.load_latest_predictions_by_market", return_value=sentinel) as mock_load:
            result = PostgresPredictionResultRepository().get_latest_by_market("us")
        mock_load.assert_called_once_with("us")
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
