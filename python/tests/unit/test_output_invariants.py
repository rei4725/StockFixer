"""予測出力の健全性チェック（出力 invariant）の単体テスト。"""

import unittest

from src.prediction.output_invariants import (
    DEGRADED_SYMBOL_RATIO_THRESHOLD,
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


if __name__ == "__main__":
    unittest.main()
