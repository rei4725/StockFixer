"""
ユニットテスト: capital_allocator

外部依存（DB・yfinance）は持たないため、モックなしで動作する。
"""

import unittest

from src.trading.capital_allocator import allocate_capital
from src.trading.types import AllocationCandidate, AllocationResult


def _make_candidate(
    symbol: str,
    diff_ratio: float,
    confidence_ratio: float,
    sector: str = "Technology",
    market: str = "us",
) -> AllocationCandidate:
    return AllocationCandidate(
        market=market,
        symbol=symbol,
        diff_ratio=diff_ratio,
        confidence_ratio=confidence_ratio,
        sector=sector,
    )


class TestAllocateCapital(unittest.TestCase):
    def test_empty_candidates_returns_empty(self):
        result = allocate_capital([])
        self.assertEqual(result, [])

    def test_single_candidate_gets_full_weight_within_cap(self):
        c = _make_candidate("AAPL", diff_ratio=0.02, confidence_ratio=1.0)
        # bull レジームで相関なし → weight = cap そのまま
        results = allocate_capital([c], max_weight_per_symbol=0.10, market_regime="bull")
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].weight, 0.10, places=5)
        self.assertEqual(results[0].symbol, "AAPL")

    def test_negative_diff_ratio_excluded(self):
        candidates = [
            _make_candidate("AAPL", diff_ratio=0.02, confidence_ratio=0.9),
            _make_candidate("MSFT", diff_ratio=-0.01, confidence_ratio=0.9),
        ]
        results = allocate_capital(candidates, min_diff_ratio=0.0)
        symbols = [r.symbol for r in results]
        self.assertIn("AAPL", symbols)
        self.assertNotIn("MSFT", symbols)

    def test_zero_diff_ratio_excluded_by_default(self):
        candidates = [
            _make_candidate("AAPL", diff_ratio=0.02, confidence_ratio=1.0),
            _make_candidate("MSFT", diff_ratio=0.0, confidence_ratio=1.0),
        ]
        results = allocate_capital(candidates)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "AAPL")

    def test_weights_sum_to_at_most_one(self):
        candidates = [
            _make_candidate("AAPL", 0.05, 0.9, sector="Tech"),
            _make_candidate("MSFT", 0.03, 0.8, sector="Tech"),
            _make_candidate("JPM", 0.04, 0.85, sector="Finance"),
        ]
        results = allocate_capital(candidates)
        total = sum(r.weight for r in results)
        self.assertLessEqual(total, 1.0 + 1e-9)

    def test_max_positions_limits_results(self):
        candidates = [
            _make_candidate(sym, 0.02, 0.8, sector="Tech")
            for sym in ["A", "B", "C", "D", "E"]
        ]
        results = allocate_capital(candidates, max_positions=3)
        self.assertLessEqual(len(results), 3)

    def test_sector_cap_limits_per_sector(self):
        candidates = [
            _make_candidate("A", 0.05, 0.9, sector="Tech"),
            _make_candidate("B", 0.04, 0.9, sector="Tech"),
            _make_candidate("C", 0.03, 0.9, sector="Tech"),
            _make_candidate("D", 0.04, 0.9, sector="Finance"),
        ]
        results = allocate_capital(candidates, max_sector_positions=2)
        tech_count = sum(1 for r in results if r.sector == "Tech")
        self.assertLessEqual(tech_count, 2)
        self.assertIn("D", [r.symbol for r in results])

    def test_bull_regime_applies_full_scale(self):
        candidates = [_make_candidate("AAPL", 0.02, 1.0)]
        bull_results = allocate_capital(candidates, market_regime="bull")
        range_results = allocate_capital(candidates, market_regime="range")
        # bull レジームのほうが range より regime_scale が大きい
        self.assertGreater(bull_results[0].regime_scale, range_results[0].regime_scale)

    def test_bear_regime_applies_minimum_scale(self):
        candidates = [_make_candidate("AAPL", 0.02, 1.0)]
        results = allocate_capital(candidates, market_regime="bear")
        self.assertAlmostEqual(results[0].regime_scale, 0.3, places=5)

    def test_high_correlation_reduces_weights(self):
        candidates = [
            _make_candidate("A", 0.04, 0.9, sector="Tech"),
            _make_candidate("B", 0.04, 0.9, sector="Finance"),
        ]
        low_corr = allocate_capital(candidates, avg_correlation=0.0)
        high_corr = allocate_capital(candidates, avg_correlation=0.9)
        total_low = sum(r.weight for r in low_corr)
        total_high = sum(r.weight for r in high_corr)
        self.assertGreater(total_low, total_high)

    def test_per_symbol_weight_cap_applied(self):
        candidates = [
            _make_candidate("A", 0.10, 1.0, sector="Tech"),
            _make_candidate("B", 0.001, 1.0, sector="Finance"),
        ]
        cap = 0.08
        results = allocate_capital(candidates, max_weight_per_symbol=cap)
        for r in results:
            self.assertLessEqual(r.weight, cap + 1e-9)

    def test_returns_sorted_by_weight_descending(self):
        candidates = [
            _make_candidate("A", 0.01, 0.5, sector="Tech"),
            _make_candidate("B", 0.05, 0.9, sector="Finance"),
            _make_candidate("C", 0.03, 0.8, sector="Health"),
        ]
        results = allocate_capital(candidates)
        weights = [r.weight for r in results]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_result_type_is_allocation_result(self):
        candidates = [_make_candidate("AAPL", 0.02, 0.9)]
        results = allocate_capital(candidates)
        self.assertIsInstance(results[0], AllocationResult)

    def test_unknown_regime_uses_default_scale(self):
        candidates = [_make_candidate("AAPL", 0.02, 1.0)]
        results = allocate_capital(candidates, market_regime="unknown_xyz")
        self.assertAlmostEqual(results[0].regime_scale, 0.6, places=5)

    def test_none_regime_uses_default_scale(self):
        candidates = [_make_candidate("AAPL", 0.02, 1.0)]
        results = allocate_capital(candidates, market_regime=None)
        self.assertAlmostEqual(results[0].regime_scale, 0.6, places=5)

    def test_all_negative_diff_ratios_returns_empty(self):
        candidates = [
            _make_candidate("A", -0.05, 0.9),
            _make_candidate("B", -0.02, 0.8),
        ]
        results = allocate_capital(candidates)
        self.assertEqual(results, [])

    def test_confidence_ratio_scales_weight_proportionally(self):
        # 同じ diff_ratio で confidence が2倍なら weight も2倍（キャップ外の範囲で検証）
        c_high = _make_candidate("HIGH", 0.03, 0.9, sector="Tech")
        c_low = _make_candidate("LOW", 0.03, 0.45, sector="Finance")
        # キャップを十分大きくして均等キャップが働かないようにする
        results = allocate_capital([c_high, c_low], max_weight_per_symbol=0.8)
        high_w = next(r.weight for r in results if r.symbol == "HIGH")
        low_w = next(r.weight for r in results if r.symbol == "LOW")
        self.assertAlmostEqual(high_w / low_w, 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
