"""
Integration Tests for prediction_pipeline - Optimal Parameters

実際の JSON ファイルから最適パラメータを読み込むテストです。
"""
import json
import os
import sys

import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class TestGetOptimalParamsIntegration:
    """get_optimal_params() の統合テスト"""

    def test_get_optimal_params_from_actual_json(self):
        """実際の JSON ファイルから最適パラメータを読み込むテスト"""
        # config/optimal_params.json が存在するか確認
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../config/optimal_params.json")
        )

        if not os.path.exists(config_path):
            pytest.skip("config/optimal_params.json が見つかりません（バックテスト最適化実行後に生成）")

        # 実際の JSON を読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)

        # 最低1つのパラメータセットが存在すること
        assert len(all_params) > 0, "JSON に最適パラメータセットが含まれていません"

    def test_optimal_params_structure(self):
        """最適パラメータの構造を検証"""
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../config/optimal_params.json")
        )

        if not os.path.exists(config_path):
            pytest.skip("config/optimal_params.json が見つかりません")

        with open(config_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)

        # 各パラメータセットの構造を検証
        for key, params in all_params.items():
            # キーがマーケット_シンボル形式であることを確認
            assert "_" in key, f"キー '{key}' が市場_シンボル形式ではありません"

            # 必須フィールドを確認
            required_fields = ["market", "symbol", "threshold", "metrics"]
            for field in required_fields:
                assert field in params, f"フィールド '{field}' が見つかりません"

            # market と symbol がキーと一致することを確認
            expected_key = f"{params['market']}_{params['symbol']}"
            assert key == expected_key, f"キーが一致しません: {key} != {expected_key}"

    def test_metrics_values_validation(self):
        """メトリクス値の妥当性を検証"""
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../config/optimal_params.json")
        )

        if not os.path.exists(config_path):
            pytest.skip("config/optimal_params.json が見つかりません")

        with open(config_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)

        for key, params in all_params.items():
            metrics = params.get("metrics", {})

            # Win Rate は 0～1 の範囲
            win_rate = metrics.get("win_rate")
            if win_rate is not None:
                assert 0 <= win_rate <= 1, f"{key}: win_rate が範囲外です ({win_rate})"

            # Profit Factor は正の数
            profit_factor = metrics.get("profit_factor")
            if profit_factor is not None:
                assert profit_factor >= 0, f"{key}: profit_factor が負の数です ({profit_factor})"

            # Num Trades は非負整数
            num_trades = metrics.get("num_trades")
            if num_trades is not None:
                assert num_trades >= 0, f"{key}: num_trades が負の数です ({num_trades})"
                assert isinstance(num_trades, int), f"{key}: num_trades が整数ではありません"

    def test_threshold_range(self):
        """閾値の範囲を検証"""
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../config/optimal_params.json")
        )

        if not os.path.exists(config_path):
            pytest.skip("config/optimal_params.json が見つかりません")

        with open(config_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)

        for key, params in all_params.items():
            threshold = params.get("threshold")

            # 閾値は通常 -1 ～ 1 の範囲
            assert threshold is not None, f"{key}: threshold が見つかりません"
            assert -1 <= threshold <= 1, f"{key}: 閾値が範囲外です ({threshold})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
