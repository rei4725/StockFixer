"""予測出力の健全性チェック（出力 invariant）の単体テスト。"""

import unittest

from src.prediction.output_invariants import (
    DEGRADED_SYMBOL_RATIO_THRESHOLD,
    PredictionRunStats,
    evaluate_output_invariants,
)
from src.prediction.types import PredictionResult

REQUESTED = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]


def make_rows(count: int, model_count: int, diff_ratio_step: float = 0.001):
    """実データと同じ形の PredictionResult を組み立てる。

    model_count を捏造した dict で代替しないこと（#615 の教訓）。
    """
    rows = []
    for i in range(count):
        price = 1000.0 + i
        diff_ratio = diff_ratio_step * (i - count / 2)
        rows.append(
            PredictionResult(
                market="jp",
                symbol=str(7000 + i),
                current_price=price,
                avg_pred_price=price * (1 + diff_ratio),
                diff_ratio=diff_ratio,
                model_count=model_count,
            )
        )
    return rows


class TestAbsoluteInvariants(unittest.TestCase):
    def test_healthy_run_has_no_violation(self):
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(705, model_count=2),
        )
        self.assertFalse(report.has_violation)
        self.assertEqual(report.violation_ids, [])

    def test_a1_fires_when_model_failed_to_load(self):
        """モデルファイル欠損。A-2 では原理的に検出できないケース。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=["UnifiedStockXGBoost"],
            output_rows=make_rows(705, model_count=1),
        )
        self.assertIn("A-1", report.violation_ids)

    def test_a2_reproduces_issue_615(self):
        """#615 の実際の状態: 2モデルともロード成功、705銘柄すべて model_count=1。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(705, model_count=1),
        )
        self.assertIn("A-2", report.violation_ids)

    def test_a2_silent_when_only_a_few_symbols_degrade(self):
        """数銘柄の片肺化は正常運用の揺らぎとして鳴らさない。"""
        rows = make_rows(100, model_count=2)
        for row in rows[:10]:
            row.model_count = 1
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
        )
        self.assertNotIn("A-2", report.violation_ids)

    def test_a2_boundary_exactly_at_threshold_fires(self):
        """縮退率ちょうど 50% は違反（>= 判定）。"""
        rows = make_rows(100, model_count=2)
        for row in rows[:50]:
            row.model_count = 1
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
        )
        self.assertIn("A-2", report.violation_ids)
        violation = next(v for v in report.violations if v.violation_id == "A-2")
        self.assertAlmostEqual(violation.observed, 0.5)
        self.assertAlmostEqual(violation.threshold, DEGRADED_SYMBOL_RATIO_THRESHOLD)

    def test_a3_reproduces_issue_612(self):
        """全銘柄がスキップされても例外にならず空リストになる経路。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=[],
        )
        self.assertIn("A-3", report.violation_ids)
        self.assertIsNone(report.stats)


class TestAsDetailsOutput(unittest.TestCase):
    def test_as_details_with_violations(self):
        """as_details() が違反を持つレポートを dict に変換できることを検証。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=["UnifiedStockXGBoost"],
            output_rows=make_rows(705, model_count=1),
        )
        # A-1 と A-2 が両方発火するはず
        self.assertTrue(report.has_violation)
        self.assertGreaterEqual(len(report.violations), 1)

        details = report.as_details()

        # トップレベルのキーを検証
        self.assertIn("compared_with_previous", details)
        self.assertIn("symbol_count", details)
        self.assertIn("median_model_count", details)
        self.assertIn("diff_ratio_stdev", details)
        self.assertIn("violations", details)

        # stats の値を検証（予測がある場合）
        self.assertIsInstance(details["symbol_count"], int)
        self.assertIsInstance(details["median_model_count"], (int, float))
        self.assertIsInstance(details["diff_ratio_stdev"], (int, float))
        self.assertFalse(details["compared_with_previous"])

        # violations が dict のリストであることを検証
        self.assertIsInstance(details["violations"], list)
        self.assertEqual(len(details["violations"]), len(report.violations))

        for violation_dict in details["violations"]:
            self.assertIn("id", violation_dict)
            self.assertIn("description", violation_dict)
            self.assertIn("observed", violation_dict)
            self.assertIn("threshold", violation_dict)
            self.assertIsInstance(violation_dict["id"], str)
            self.assertIsInstance(violation_dict["description"], str)
            self.assertIsInstance(violation_dict["observed"], (int, float))
            self.assertIsInstance(violation_dict["threshold"], (int, float))

    def test_as_details_with_stats_none(self):
        """as_details() が stats=None のレポートで適切なデフォルト値を返すことを検証。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=[],
        )
        # A-3 が発火し、stats は None になる
        self.assertIn("A-3", report.violation_ids)
        self.assertIsNone(report.stats)

        details = report.as_details()

        # stats=None のときは 0/0.0 がセットされるべき
        self.assertEqual(details["symbol_count"], 0)
        self.assertEqual(details["median_model_count"], 0.0)
        self.assertEqual(details["diff_ratio_stdev"], 0.0)

        # violations は A-3 のみ
        self.assertEqual(len(details["violations"]), 1)
        self.assertEqual(details["violations"][0]["id"], "A-3")


class TestRegressionInvariants(unittest.TestCase):
    def _healthy_previous(self) -> PredictionRunStats:
        return PredictionRunStats(symbol_count=705, median_model_count=2.0, diff_ratio_stdev=0.01)

    def test_b1_fires_on_large_symbol_drop(self):
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(500, model_count=2),
            previous_stats=self._healthy_previous(),
        )
        self.assertIn("B-1", report.violation_ids)
        self.assertTrue(report.compared_with_previous)

    def test_b1_silent_on_symbol_increase(self):
        """銘柄追加は正常。片側判定であることを確認する。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(900, model_count=2),
            previous_stats=self._healthy_previous(),
        )
        self.assertNotIn("B-1", report.violation_ids)

    def test_b1_boundary_exactly_20_percent_drop_fires(self):
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(564, model_count=2),  # 705 * 0.8
            previous_stats=self._healthy_previous(),
        )
        self.assertIn("B-1", report.violation_ids)

    def test_b2_fires_when_median_model_count_drops(self):
        rows = make_rows(705, model_count=2)
        for row in rows[:400]:
            row.model_count = 1
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
            previous_stats=self._healthy_previous(),
        )
        self.assertIn("B-2", report.violation_ids)

    def test_b3_fires_when_stdev_shrinks_by_half(self):
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(705, model_count=2, diff_ratio_step=0.00001),
            previous_stats=self._healthy_previous(),
        )
        self.assertIn("B-3", report.violation_ids)

    def test_b3_fires_on_zero_stdev_without_previous(self):
        """全銘柄が同じ予測値。前回統計が無くても違反。"""
        rows = make_rows(705, model_count=2)
        for row in rows:
            row.diff_ratio = 0.005
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
        )
        self.assertIn("B-3", report.violation_ids)

    def test_regression_skipped_without_previous_stats(self):
        """前回統計が無ければ急変ルールは評価されない。"""
        rows = make_rows(1, model_count=2)
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
        )
        self.assertFalse(report.compared_with_previous)
        for vid in ("B-1", "B-2"):
            self.assertNotIn(vid, report.violation_ids)


if __name__ == "__main__":
    unittest.main()
