"""prediction_pipeline モジュールのユニットテスト"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestGetOptimalParams(unittest.TestCase):
    """get_optimal_params のテスト（JSON ファイルをモック）"""

    def test_returns_empty_when_file_not_exists(self):
        from src.services.prediction_pipeline import get_optimal_params

        with tempfile.TemporaryDirectory() as tmp:
            # json ファイルを作らない
            nonexistent = Path(tmp) / "config" / "optimal_params.json"
            with patch("src.services.prediction_pipeline.Path") as MockPath:
                # chaining: Path(__file__).parents[2] / "config"
                mock_chain = MockPath.return_value
                chained = mock_chain.parents.__getitem__.return_value
                chained.__truediv__.return_value.__truediv__.return_value = nonexistent
                result = get_optimal_params("us", "AAPL")
        self.assertEqual(result, {})

    def test_returns_matching_params(self):
        from src.services.prediction_pipeline import get_optimal_params

        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            json_path = config_dir / "optimal_params.json"
            data = {"us_AAPL": {"threshold": 0.02, "metrics": {"sharpe_ratio": 1.5}}}
            json_path.write_text(json.dumps(data), encoding="utf-8")

            with patch("src.services.prediction_pipeline.Path") as MockPath:
                parents_mock = MockPath.return_value.parents.__getitem__.return_value
                parents_mock.__truediv__.return_value.__truediv__.return_value = json_path
                result = get_optimal_params("us", "AAPL")
        self.assertEqual(result.get("threshold"), 0.02)

    def test_returns_empty_when_symbol_missing(self):
        from src.services.prediction_pipeline import get_optimal_params

        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            json_path = config_dir / "optimal_params.json"
            data = {"us_GOOG": {"threshold": 0.01}}
            json_path.write_text(json.dumps(data), encoding="utf-8")

            with patch("src.services.prediction_pipeline.Path") as MockPath:
                parents_mock = MockPath.return_value.parents.__getitem__.return_value
                parents_mock.__truediv__.return_value.__truediv__.return_value = json_path
                result = get_optimal_params("us", "AAPL")
        self.assertEqual(result, {})

    def test_returns_empty_on_invalid_json(self):
        from src.services.prediction_pipeline import get_optimal_params

        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            json_path = config_dir / "optimal_params.json"
            json_path.write_text("not-json", encoding="utf-8")

            with patch("src.services.prediction_pipeline.Path") as MockPath:
                parents_mock = MockPath.return_value.parents.__getitem__.return_value
                parents_mock.__truediv__.return_value.__truediv__.return_value = json_path
                result = get_optimal_params("us", "AAPL")
        self.assertEqual(result, {})


class TestFindModelFiles(unittest.TestCase):
    """find_model_files のテスト（実ディレクトリを作成して glob を検証）"""

    def _setup_model_dir(self, tmp_dir, entries):
        """entries: [(市場_銘柄, モデル名)] のリスト"""
        for folder, model_name in entries:
            folder_path = os.path.join(tmp_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            open(os.path.join(folder_path, model_name), "w").close()

    def test_finds_matching_models(self):
        from src.services.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, [("us_AAPL", "StockXGBoostModel.joblib")])
            result = find_model_files(model_root=tmp)
        self.assertEqual(len(result), 1)
        market, symbol, path = result[0]
        self.assertEqual(market, "us")
        self.assertEqual(symbol, "AAPL")

    def test_returns_empty_when_no_match(self):
        from src.services.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, [("us_AAPL", "OtherModel.joblib")])
            result = find_model_files(model_root=tmp, model_name="StockXGBoostModel.joblib")
        self.assertEqual(result, [])

    def test_finds_multiple_symbols(self):
        from src.services.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(
                tmp,
                [
                    ("us_AAPL", "StockXGBoostModel.joblib"),
                    ("jp_7203", "StockXGBoostModel.joblib"),
                ],
            )
            result = find_model_files(model_root=tmp)
        self.assertEqual(len(result), 2)
        markets = {r[0] for r in result}
        self.assertIn("us", markets)
        self.assertIn("jp", markets)

    def test_returns_correct_path(self):
        from src.services.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, [("us_AAPL", "StockXGBoostModel.joblib")])
            result = find_model_files(model_root=tmp)
            _, _, path = result[0]
            self.assertTrue(os.path.exists(path))

    def test_custom_model_name(self):
        from src.services.prediction_pipeline import find_model_files

        with tempfile.TemporaryDirectory() as tmp:
            self._setup_model_dir(tmp, [("us_AAPL", "StockLightGBMModel.joblib")])
            result = find_model_files(model_root=tmp, model_name="StockLightGBMModel.joblib")
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
