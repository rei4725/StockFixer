"""A-4: 市場ごとに予想が同じ向きへ寄った形を見張る不変条件のテスト（#1814）。"""

import unittest

from src.prediction import output_invariants
from src.prediction.output_invariants import evaluate_output_invariants
from src.prediction.types import PredictionResult

_MODELS = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]


def _rows(market: str, up: int, down: int) -> list[PredictionResult]:
    """up 件を上げ、down 件を下げの予想にした行を作る。値は銘柄ごとに変える。"""
    rows = []
    for i in range(up + down):
        sign = 1.0 if i < up else -1.0
        rows.append(
            PredictionResult(
                market=market,
                symbol=f"{market}{i}",
                current_price=1000.0,
                avg_pred_price=1000.0,
                diff_ratio=sign * 0.001 * (i + 1),
                model_count=len(_MODELS),
            )
        )
    return rows


def _evaluate(rows: list[PredictionResult]):
    return evaluate_output_invariants(_MODELS, _MODELS, rows)


class TestSameDirectionInvariant(unittest.TestCase):
    def test_96_percent_up_fires_a4(self):
        report = _evaluate(_rows("jp", up=96, down=4))
        self.assertIn("A-4", report.violation_ids)
        violation = next(v for v in report.violations if v.violation_id == "A-4")
        self.assertAlmostEqual(violation.observed, 0.96)
        self.assertEqual(violation.threshold, output_invariants.SAME_DIRECTION_RATIO_THRESHOLD)
        self.assertIn("jp", violation.description)

    def test_96_percent_down_fires_a4(self):
        report = _evaluate(_rows("us", up=4, down=96))
        self.assertIn("A-4", report.violation_ids)

    def test_90_percent_up_does_not_fire(self):
        report = _evaluate(_rows("jp", up=90, down=10))
        self.assertNotIn("A-4", report.violation_ids)

    def test_few_symbols_do_not_fire(self):
        count = output_invariants.SAME_DIRECTION_MIN_SYMBOLS - 1
        report = _evaluate(_rows("jp", up=count, down=0))
        self.assertNotIn("A-4", report.violation_ids)

    def test_min_symbols_all_up_fires(self):
        count = output_invariants.SAME_DIRECTION_MIN_SYMBOLS
        report = _evaluate(_rows("jp", up=count, down=0))
        self.assertIn("A-4", report.violation_ids)

    def test_judged_per_market(self):
        """片方の市場だけが寄った日は、その市場だけを1件として出す。"""
        rows = _rows("jp", up=99, down=1) + _rows("us", up=50, down=50)
        report = _evaluate(rows)
        a4 = [v for v in report.violations if v.violation_id == "A-4"]
        self.assertEqual(len(a4), 1)
        self.assertIn("jp", a4[0].description)

    def test_both_markets_fire_separately(self):
        rows = _rows("jp", up=99, down=1) + _rows("us", up=99, down=1)
        report = _evaluate(rows)
        self.assertEqual(report.violation_ids.count("A-4"), 2)

    def test_a4_appears_in_details(self):
        report = _evaluate(_rows("jp", up=100, down=0))
        ids = [v["id"] for v in report.as_details()["violations"]]
        self.assertIn("A-4", ids)

    def test_balanced_run_has_no_violation(self):
        report = _evaluate(_rows("jp", up=50, down=50))
        self.assertEqual(report.violation_ids, [])


if __name__ == "__main__":
    unittest.main()
