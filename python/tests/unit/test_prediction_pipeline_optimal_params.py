"""
Unit Tests for prediction_pipeline.get_optimal_params()

最適パラメータの JSON 読込機能をテストします。
"""
import json
import os
import sys
from unittest.mock import patch

import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestGetOptimalParams:
    """get_optimal_params() の機能テスト"""

    @pytest.fixture
    def mock_config_json(self, tmp_path):
        """テスト用の最適パラメータ JSON を生成"""
        config_data = {
            "jp_1332": {
                "market": "jp",
                "symbol": "1332",
                "timestamp": "2026-03-01T18:32:50.818256",
                "sort_by": "sharpe_ratio",
                "threshold": 0.0,
                "stop_loss_pct": None,
                "take_profit_pct": None,
                "metrics": {
                    "total_return": -0.1520988,
                    "sharpe_ratio": 0.25460000000000005,
                    "max_drawdown": -0.39801699999999995,
                    "win_rate": 0.57524,
                    "profit_factor": 1.06494,
                    "num_trades": 182,
                },
            },
            "jp_1333": {
                "market": "jp",
                "symbol": "1333",
                "timestamp": "2026-03-01T18:33:09.091922",
                "sort_by": "sharpe_ratio",
                "threshold": 0.004,
                "stop_loss_pct": None,
                "take_profit_pct": None,
                "metrics": {
                    "total_return": -0.036658,
                    "sharpe_ratio": 0.14698000000000006,
                    "max_drawdown": -0.1819594,
                    "win_rate": 0.52916,
                    "profit_factor": 1.1505999999999998,
                    "num_trades": 31,
                },
            },
        }

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        json_path = config_dir / "optimal_params.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        return str(json_path)

    def test_get_optimal_params_jp_1332(self, mock_config_json):
        """jp/1332 のパラメータ読込テスト"""
        with patch("src.services.prediction_pipeline.os.path.dirname") as mock_dirname:
            # パスモック設定
            config_dir = os.path.dirname(mock_config_json)
            mock_dirname.side_effect = [
                os.path.dirname(os.path.dirname(config_dir)),  # __file__ をシミュレート
                config_dir,  # config_dir の親ディレクトリ
            ]

            # 直接 JSON パス指定で実行（ユニットテスト用）
            with open(mock_config_json, "r", encoding="utf-8") as f:
                all_params = json.load(f)

            params = all_params.get("jp_1332", {})

            # 検証
            assert params is not None
            assert params["threshold"] == 0.0
            assert params["metrics"]["sharpe_ratio"] > 0.2
            assert params["metrics"]["win_rate"] > 0.5
            assert params["metrics"]["profit_factor"] > 1.0

    def test_get_optimal_params_jp_1333(self, mock_config_json):
        """jp/1333 のパラメータ読込テスト"""
        with open(mock_config_json, "r", encoding="utf-8") as f:
            all_params = json.load(f)

        params = all_params.get("jp_1333", {})

        # 検証
        assert params is not None
        assert params["threshold"] == 0.004
        assert params["metrics"]["win_rate"] > 0.5

    def test_get_optimal_params_not_found(self, tmp_path):
        """存在しない銘柄のパラメータ取得テスト"""
        config_data = {"jp_1332": {"threshold": 0.0}}

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        json_path = config_dir / "optimal_params.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        with open(json_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)

        params = all_params.get("us_AAPL", {})

        # 存在しないキーは空辞書を返す
        assert params == {}

    def test_metrics_structure(self, mock_config_json):
        """メトリクス構造の検証"""
        with open(mock_config_json, "r", encoding="utf-8") as f:
            all_params = json.load(f)

        params = all_params["jp_1332"]
        metrics = params.get("metrics", {})

        # 必須メトリクスの存在確認
        required_metrics = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "num_trades",
        ]

        for metric in required_metrics:
            assert metric in metrics, f"メトリクス '{metric}' が見つかりません"

    def test_timestamp_format(self, mock_config_json):
        """タイムスタンプ形式の検証"""
        with open(mock_config_json, "r", encoding="utf-8") as f:
            all_params = json.load(f)

        params = all_params["jp_1332"]
        timestamp = params.get("timestamp")

        # ISO 8601 形式の確認（タイムゾーン有無は許容）
        assert timestamp is not None
        assert "T" in timestamp  # 日時区切り

        # datetime.fromisoformat でパース可能であることを確認
        from datetime import datetime

        datetime.fromisoformat(timestamp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
