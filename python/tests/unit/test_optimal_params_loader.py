"""
ユニットテスト: optimal_params_loader

tmp_path を使ったファイルベースの単体テスト。
外部ファイルシステム以外の依存はなし。
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch


class TestGetOptimalParams(unittest.TestCase):
    """get_optimal_params() のファイルベーステスト"""

    def _get_fn(self):
        from src.strategy.optimal_params_loader import get_optimal_params

        return get_optimal_params

    def test_returns_params_when_file_and_key_exist(self, tmp_path=None):
        """ファイルが存在し、対象キーがある場合はパラメータ辞書を返す"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "optimal_params.json"
            data = {"jp_7203": {"threshold": 0.01, "stop_loss_pct": 0.03, "take_profit_pct": 0.07}}
            path.write_text(json.dumps(data), encoding="utf-8")

            get_optimal_params = self._get_fn()
            with patch("src.strategy.optimal_params_loader.OPTIMAL_PARAMS_PATH", path):
                result = get_optimal_params("jp", "7203")

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["threshold"], 0.01)
        self.assertAlmostEqual(result["stop_loss_pct"], 0.03)
        self.assertAlmostEqual(result["take_profit_pct"], 0.07)

    def test_returns_none_when_file_not_exists(self):
        """ファイルが存在しない場合は None を返す"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            missing_path = Path(d) / "does_not_exist.json"

            get_optimal_params = self._get_fn()
            with patch("src.strategy.optimal_params_loader.OPTIMAL_PARAMS_PATH", missing_path):
                result = get_optimal_params("jp", "7203")

        self.assertIsNone(result)

    def test_returns_none_when_key_not_in_json(self):
        """ファイルは存在するが対象キーがない場合は None を返す"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "optimal_params.json"
            data = {"us_AAPL": {"threshold": 0.02}}
            path.write_text(json.dumps(data), encoding="utf-8")

            get_optimal_params = self._get_fn()
            with patch("src.strategy.optimal_params_loader.OPTIMAL_PARAMS_PATH", path):
                result = get_optimal_params("jp", "9999")

        self.assertIsNone(result)

    def test_returns_none_when_json_is_invalid(self):
        """ファイルが不正な JSON の場合は None を返す（例外を外に出さない）"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "optimal_params.json"
            path.write_text("{ invalid json !!!}", encoding="utf-8")

            get_optimal_params = self._get_fn()
            with patch("src.strategy.optimal_params_loader.OPTIMAL_PARAMS_PATH", path):
                result = get_optimal_params("jp", "7203")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
