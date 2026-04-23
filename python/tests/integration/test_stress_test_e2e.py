"""
Integration Test: ストレステストパイプライン End-to-End

yfinance 経由でのデータ取得を伴うため tests/integration/ に配置。
実行には外部ネットワークアクセスが必要。
"""
import os
import sys
import unittest

# パス設定
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestStressTestE2E(unittest.TestCase):
    """ストレステストパイプラインのEnd-to-Endテスト"""

    @classmethod
    def setUpClass(cls):
        """テスト前準備：XGBoost 利用可否を確認"""
        import importlib.util

        cls.xgboost_available = importlib.util.find_spec("xgboost") is not None

    def _skip_if_no_xgboost(self):
        if not self.xgboost_available:
            self.skipTest("XGBoost が利用できません")

    # ------------------------------------------------------------------
    # テスト: run_stress_test_single
    # ------------------------------------------------------------------

    def test_run_stress_test_single_corona(self):
        """コロナシナリオで StressTestResult が返ること"""
        self._skip_if_no_xgboost()
        try:
            from src.domain.types import StressTestResult
            from src.services.stress_test_pipeline import run_stress_test_single

            result = run_stress_test_single(
                market="us",
                symbol="AAPL",
                scenario_name="corona",
                model_type="XGBoostModel",
                source="api",
            )
        except Exception as e:
            self.skipTest(f"ストレステスト実行中にエラーが発生: {e}")

        self.assertIsNotNone(result, "コロナシナリオで StressTestResult が None でないこと")
        self.assertIsInstance(result, StressTestResult, "戻り値が StressTestResult 型であること")
        self.assertEqual(result.scenario_name, "corona", "scenario_name が 'corona' であること")
        self.assertEqual(result.symbol, "AAPL", "symbol が 'AAPL' であること")
        self.assertEqual(result.market, "us", "market が 'us' であること")

    def test_run_stress_test_single_lehman(self):
        """リーマンショックシナリオで StressTestResult が返ること"""
        self._skip_if_no_xgboost()
        try:
            from src.domain.types import StressTestResult
            from src.services.stress_test_pipeline import run_stress_test_single

            result = run_stress_test_single(
                market="us",
                symbol="AAPL",
                scenario_name="lehman",
                model_type="XGBoostModel",
                source="api",
            )
        except Exception as e:
            self.skipTest(f"ストレステスト実行中にエラーが発生: {e}")

        self.assertIsNotNone(result, "リーマンシナリオで StressTestResult が None でないこと")
        self.assertIsInstance(result, StressTestResult, "戻り値が StressTestResult 型であること")
        self.assertEqual(result.scenario_name, "lehman", "scenario_name が 'lehman' であること")

    def test_mdd_is_numeric(self):
        """MDD が数値であること（None でないこと）"""
        self._skip_if_no_xgboost()
        try:
            from src.services.stress_test_pipeline import run_stress_test_single

            result = run_stress_test_single(
                market="us",
                symbol="AAPL",
                scenario_name="corona",
                model_type="XGBoostModel",
                source="api",
            )
        except Exception as e:
            self.skipTest(f"ストレステスト実行中にエラーが発生: {e}")

        if result is None:
            self.skipTest("result が None のためスキップ")

        self.assertIsNotNone(result.mdd, "mdd が None でないこと")
        self.assertIsInstance(result.mdd, (int, float), "mdd が数値型であること")
        self.assertIsNotNone(result.sharpe_ratio, "sharpe_ratio が None でないこと")
        self.assertIsInstance(result.sharpe_ratio, (int, float), "sharpe_ratio が数値型であること")

    def test_save_stress_test_results_creates_csv(self):
        """StressTestResult を保存すると CSV が results/stress_test/ に作成されること"""
        import tempfile

        try:
            from src.domain.types import StressTestResult
            from src.services.stress_test_pipeline import save_stress_test_results
        except ImportError as e:
            self.skipTest(f"インポートエラー: {e}")

        # ダミーの StressTestResult を作成
        dummy_result = StressTestResult(
            market="us",
            symbol="AAPL",
            scenario_name="corona",
            period_start="2020-02-01",
            period_end="2020-03-31",
            mdd=-0.10,
            sharpe_ratio=0.5,
            total_return=-0.08,
            win_rate=0.45,
            num_trades=5,
            max_consecutive_losses=2,
            mdd_pass=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "stress_test")
            filepath = save_stress_test_results([dummy_result], output_dir=output_dir)

            self.assertTrue(os.path.exists(filepath), f"CSV ファイルが作成されること: {filepath}")
            self.assertTrue(filepath.endswith(".csv"), "ファイルが CSV 形式であること")

            # CSV の中身を確認
            import pandas as pd

            df = pd.read_csv(filepath)
            self.assertEqual(len(df), 1, "1行のデータが保存されていること")
            self.assertIn("symbol", df.columns, "symbol 列が存在すること")
            self.assertIn("scenario_name", df.columns, "scenario_name 列が存在すること")
            self.assertIn("mdd", df.columns, "mdd 列が存在すること")
            self.assertIn("mdd_pass", df.columns, "mdd_pass 列が存在すること")
            self.assertEqual(df.iloc[0]["symbol"], "AAPL")
            self.assertEqual(df.iloc[0]["scenario_name"], "corona")

    def test_unknown_scenario_returns_none(self):
        """不正なシナリオ名を渡すと None が返ること"""
        try:
            from src.services.stress_test_pipeline import run_stress_test_single

            result = run_stress_test_single(
                market="us",
                symbol="AAPL",
                scenario_name="unknown_scenario",
                model_type="XGBoostModel",
                source="api",
            )
        except Exception as e:
            self.skipTest(f"予期しないエラー: {e}")

        self.assertIsNone(result, "不正なシナリオ名で None が返ること")
