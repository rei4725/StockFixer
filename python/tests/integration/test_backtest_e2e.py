"""
Integration Test: バックテストパイプライン End-to-End

DuckDB から特徴量取得 → モデル学習 → バックテスト実行
完全なフロー検証。
"""
import unittest
import sys
import os

import pandas as pd
import duckdb

# パス設定
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestBacktestPipelineE2E(unittest.TestCase):
    """バックテストパイプラインのEnd-to-Endテスト"""

    @classmethod
    def setUpClass(cls):
        """テスト前準備：環境確認"""
        try:
            import xgboost
            cls.xgboost_available = True
        except ImportError:
            cls.xgboost_available = False

    def test_duckdb_tables_exist(self):
        """DuckDBテーブルが存在することを確認"""
        try:
            conn = duckdb.connect("data/stockfixer.duckdb", read_only=True)
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
            table_names = [t[0] for t in tables]

            self.assertIn(
                "stock_features",
                table_names,
                "stock_features テーブルが存在すること",
            )
            self.assertIn(
                "market_data_raw",
                table_names,
                "market_data_raw テーブルが存在すること",
            )

            conn.close()
        except Exception as e:
            self.skipTest(f"DuckDB にアクセスできません: {e}")

    def test_jp_7203_data_exists(self):
        """JP/7203のデータが取得可能なことを確認"""
        try:
            conn = duckdb.connect("data/stockfixer.duckdb", read_only=True)

            result = conn.execute(
                "SELECT COUNT(*) FROM stock_features WHERE symbol='7203'"
            ).fetchone()
            count = result[0] if result else 0

            self.assertGreater(
                count,
                0,
                "stock_features (symbol=7203) に最低1行以上のデータが存在すること",
            )

            conn.close()
        except Exception as e:
            self.skipTest(f"JP/7203 データ確認に失敗: {e}")

    @unittest.skipIf(
        not __import__("importlib").util.find_spec("xgboost"),
        "XGBoost not available"
    )
    def test_backtest_single_runs_without_error(self):
        """単一期間バックテストが実行可能なことを確認"""
        try:
            from src.services.backtest_pipeline import (
                run_backtest_single,
                print_backtest_metrics,
            )

            result_df, metrics, wf_df = run_backtest_single(
                market="jp",
                symbol="7203",
                model_type="XGBoostModel",
                model_name="TestBacktestModel",
                task_name="return_regression",
                threshold=0.0,
                source="file",
                initial_cash=1_000_000,
                fee_rate=0.001,
                slippage=0.0,
                stop_loss_pct=None,
                take_profit_pct=None,
                position_sizing="full",
                position_fraction=0.5,
                ensemble=False,
                start_date=None,
                end_date=None,
                train_ratio=0.8,
            )

            # メトリクスが存在し、基本的な構造を持つことを確認
            self.assertIsNotNone(metrics, "メトリクスが None でないこと")
            self.assertIn("final_cash", metrics, "final_cash メトリクスが存在すること")
            self.assertIn("total_return", metrics, "total_return メトリクスが存在すること")
            self.assertIn("sharpe_ratio", metrics, "sharpe_ratio メトリクスが存在すること")

            print(f"\n[Integration Test Result] バックテスト実行成功")
            print_backtest_metrics(metrics, label="jp_7203_single")

        except Exception as e:
            self.fail(f"バックテスト実行に失敗: {e}")

    @unittest.skipIf(
        not __import__("importlib").util.find_spec("xgboost"),
        "XGBoost not available"
    )
    def test_backtest_walk_forward_runs_without_error(self):
        """Walk-Forward バックテストが実行可能なことを確認"""
        try:
            from src.services.backtest_pipeline import run_backtest_walk_forward

            _, _, wf_df = run_backtest_walk_forward(
                market="jp",
                symbol="7203",
                model_type="XGBoostModel",
                model_name="TestWalkForwardModel",
                task_name="return_regression",
                threshold=0.0,
                source="file",
                n_splits=3,  # テスト用に分割数を少なく
                initial_cash=1_000_000,
                fee_rate=0.001,
                slippage=0.0,
                stop_loss_pct=None,
                take_profit_pct=None,
                position_sizing="full",
                position_fraction=0.5,
                ensemble=False,
            )

            # Walk-Forward 結果が存在することを確認
            self.assertIsNotNone(wf_df, "Walk-Forward 結果が None でないこと")
            self.assertGreater(
                len(wf_df), 0, "Walk-Forward 結果に最低1行以上のデータがあること"
            )

            # 重要なメトリクス列が存在することを確認
            expected_cols = ["fold", "val_start", "val_end", "total_return", "sharpe_ratio"]
            for col in expected_cols:
                self.assertIn(col, wf_df.columns, f"{col} 列が存在すること")

            print(f"\n[Integration Test Result] Walk-Forward バックテスト実行成功")
            print(f"  分割数: {len(wf_df)}")

        except Exception as e:
            self.fail(f"Walk-Forward バックテスト実行に失敗: {e}")


if __name__ == "__main__":
    unittest.main()
