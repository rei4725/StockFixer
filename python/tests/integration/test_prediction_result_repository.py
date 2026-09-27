"""結合テスト: PostgresPredictionResultRepository を実 DB（prediction_results）に対して検証する。"""

import unittest

from src.infrastructure.persistence.prediction_result_repository import (
    PostgresPredictionResultRepository,
)
from src.prediction.db import save_prediction_results
from src.prediction.types import PredictionResult


def _result(symbol: str, diff_ratio, market: str = "us") -> PredictionResult:
    return PredictionResult(
        market=market,
        symbol=symbol,
        current_price=100.0,
        avg_pred_price=100.0 if diff_ratio is None else 100.0 * (1 + diff_ratio),
        diff_ratio=diff_ratio,
        model_count=2,
    )


class TestPostgresPredictionResultRepository(unittest.TestCase):
    def test_reads_what_prediction_bc_saved(self):
        save_prediction_results(
            "20990101_090000",
            [_result("ZZA", 0.01), _result("ZZB", 0.03), _result("ZZC", None)],
        )
        save_prediction_results(
            "20990101_090000",
            [_result("ZZJ", 0.02, market="jp"), _result("ZZK", 0.04, market="jp")],
        )
        repo = PostgresPredictionResultRepository()

        self.assertEqual(repo.latest_timestamp(), "20990101_090000")
        self.assertIn("us", repo.markets_at("20990101_090000"))

        top = repo.results_at(predicted_at="20990101_090000", market="jp", top_n=1)
        self.assertEqual(list(top["symbol"]), ["ZZK"])

        latest = repo.get_latest_by_market("us")
        symbols = list(latest["symbol"])
        # diff_ratio が NULL の行は除外され、diff_ratio 降順に並ぶ
        self.assertNotIn("ZZC", symbols)
        self.assertLess(symbols.index("ZZB"), symbols.index("ZZA"))
        self.assertNotIn("ZZJ", symbols)
        self.assertIn("confluence_score", latest.columns)

    def test_null_diff_ratio_is_ranked_last(self):
        """diff_ratio が NULL の行は上位にも全件の先頭にも来ない（#748）。

        Postgres の DESC は既定で NULLS FIRST。DuckDB 時代（NULLS LAST）と同じ並びに揃える。
        """
        save_prediction_results(
            "20990102_090000",
            [_result("NLA", 0.01), _result("NLB", None), _result("NLC", 0.03)],
        )
        repo = PostgresPredictionResultRepository()

        top = repo.results_at(predicted_at="20990102_090000", market="us", top_n=2)
        self.assertEqual(list(top["symbol"]), ["NLC", "NLA"])

        every = repo.results_at(predicted_at="20990102_090000", market="us")
        self.assertEqual(list(every["symbol"]), ["NLC", "NLA", "NLB"])

        worst = repo.results_at(predicted_at="20990102_090000", market="us", worst_n=2)
        self.assertEqual(list(worst["symbol"]), ["NLA", "NLC"])


if __name__ == "__main__":
    unittest.main()
