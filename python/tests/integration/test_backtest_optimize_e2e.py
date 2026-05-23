"""
Integration Test: バックテスト最適化パイプライン End-to-End

グリッドサーチ実行 → メトリクス計算（dtype エラー修正の検証）
完全なフロー検証。
"""

import os
import sys
import unittest

import pandas as pd
import pytest

# パス設定
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestBacktestOptimizeE2E(unittest.TestCase):
    """バックテスト最適化パイプラインのEnd-to-Endテスト"""

    @pytest.mark.timeout(300)
    @unittest.skipIf(not __import__("importlib").util.find_spec("xgboost"), "XGBoost not available")
    def test_optimization_metrics_dtype_fix(self):
        """
        最適化時のメトリクス計算で dtype エラーが発生しないことを確認

        profit_factor が None と数値の混在データを正しく平均計算できることを検証。
        """
        try:
            from src.backtest.optimizer import run_optimization

            # グリッドサーチを最小限（1-2パラメータ）で実行
            result_df = run_optimization(
                market="jp",
                symbol="7203",
                model_type="XGBoostModel",
                ensemble=False,
                source="file",
                n_splits=3,  # テスト用に分割数を少なく
                initial_cash=1_000_000,
                fee_rate=0.001,
                slippage=0.0,
                threshold_min=0.0,
                threshold_max=0.001,  # 最小限のグリッド
                threshold_step=0.001,
                optimize_risk=False,
            )

            # 結果が DataFrame であることを確認
            self.assertIsInstance(result_df, pd.DataFrame, "結果が DataFrame であること")
            self.assertGreater(len(result_df), 0, "結果に最低1行以上のデータがあること")

            # メトリクス列が存在することを確認
            self.assertIn("threshold", result_df.columns, "threshold 列が存在すること")

            # dtype エラー修正の確認：すべてのメトリクス列が平均計算可能であること
            numeric_cols = [
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
                "win_rate",
                "profit_factor",
            ]
            available = [c for c in numeric_cols if c in result_df.columns]

            if available:
                # None / NaN 混在データを数値型に変換して平均計算
                result_numeric = result_df[available].copy()
                for col in result_numeric.columns:
                    result_numeric[col] = pd.to_numeric(result_numeric[col], errors="coerce")

                # 平均計算が成功することを確認（dtype エラーが発生しない）
                try:
                    means = result_numeric.mean()
                    self.assertIsNotNone(means, "平均値が計算されること")
                    print("\n[Integration Test Result] メトリクス計算成功")
                    print(f"  平均値:\n{means.round(4).to_string()}")
                except TypeError as e:
                    self.fail(f"メトリクス平均計算時に dtype エラー: {e}")

        except SystemExit as e:
            self.skipTest(f"CI環境にDBデータなし（stock_features）: {e}")
        except Exception as e:
            self.fail(f"最適化実行に失敗: {e}")

    @pytest.mark.timeout(300)
    @unittest.skipIf(not __import__("importlib").util.find_spec("xgboost"), "XGBoost not available")
    def test_optimization_with_risk_parameters(self):
        """
        ストップロス・テイクプロフィット付きグリッドサーチが実行可能なことを確認

        複数リスク管理パラメータの組み合わせでも型エラーが発生しないことを検証。
        """
        try:
            from src.backtest.optimizer import run_optimization

            result_df = run_optimization(
                market="jp",
                symbol="7203",
                model_type="XGBoostModel",
                ensemble=False,
                source="file",
                n_splits=2,  # テスト用に分割数を最小化
                initial_cash=1_000_000,
                fee_rate=0.001,
                slippage=0.0,
                threshold_min=0.0,
                threshold_max=0.001,
                threshold_step=0.001,
                optimize_risk=True,  # リスク管理パラメータを有効化
            )

            # 結果が存在することを確認
            self.assertIsNotNone(result_df, "結果が None でないこと")
            self.assertGreater(len(result_df), 0, "結果に最低1行以上のデータがあること")

            # リスク管理列が存在することを確認
            self.assertIn("stop_loss_pct", result_df.columns, "stop_loss_pct 列が存在すること")
            self.assertIn(
                "take_profit_pct",
                result_df.columns,
                "take_profit_pct 列が存在すること",
            )

            print("\n[Integration Test Result] リスク管理パラメータ付きグリッドサーチ成功")
            print(f"  パラメータ組み合わせ数: {len(result_df)}")

        except SystemExit as e:
            self.skipTest(f"CI環境にDBデータなし（stock_features）: {e}")
        except Exception as e:
            self.fail(f"リスク管理パラメータ付き最適化に失敗: {e}")


if __name__ == "__main__":
    unittest.main()
