"""
ユニットテスト: capital_allocator (R-301)

DuckDB・yfinance・market_regime はすべて MagicMock で差し替える。
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.trading.capital_allocator import (
    AllocationResult,
    _apply_correlation_penalty,
    _detect_regime,
    _load_confidence_map,
    compute_portfolio_allocation,
)

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _make_candidates(
    symbols: list[str], diff_ratios: list[float], market: str = "jp"
) -> list[dict]:
    return [
        {"market": market, "symbol": sym, "diff_ratio": dr} for sym, dr in zip(symbols, diff_ratios)
    ]


# ---------------------------------------------------------------------------
# AllocationResult dataclass
# ---------------------------------------------------------------------------


class TestAllocationResult(unittest.TestCase):
    def test_fields_exist(self):
        r = AllocationResult(
            market="jp",
            symbol="7203",
            weight=0.08,
            position_size_ratio=0.8,
            signal_strength=0.02,
            confidence=0.65,
            regime="bull",
            sector="Automotive",
            allocation_reason="signal=0.0200, confidence=0.65",
        )
        self.assertEqual(r.market, "jp")
        self.assertEqual(r.symbol, "7203")
        self.assertAlmostEqual(r.weight, 0.08)
        self.assertEqual(r.regime, "bull")
        self.assertEqual(r.allocated_capital, 0.0)  # デフォルト

    def test_default_sector_is_none(self):
        r = AllocationResult(
            market="jp",
            symbol="9984",
            weight=0.05,
            position_size_ratio=0.5,
            signal_strength=0.01,
            confidence=0.6,
            regime="range",
        )
        self.assertIsNone(r.sector)
        self.assertEqual(r.allocation_reason, "")


# ---------------------------------------------------------------------------
# _apply_correlation_penalty
# ---------------------------------------------------------------------------


class TestApplyCorrelationPenalty(unittest.TestCase):
    def test_no_penalty_below_threshold(self):
        symbols = ["A", "B"]
        weights = {"A": 0.5, "B": 0.5}
        # 相関 0.0 → ペナルティなし
        returns = pd.DataFrame(
            {"A": np.random.RandomState(0).randn(20), "B": np.random.RandomState(1).randn(20)}
        )
        result = _apply_correlation_penalty(symbols, weights, returns)
        self.assertAlmostEqual(result["A"], 0.5)
        self.assertAlmostEqual(result["B"], 0.5)

    def test_penalty_applied_for_high_corr(self):
        symbols = ["A", "B"]
        weights = {"A": 0.5, "B": 0.6}
        base = np.cumsum(np.random.RandomState(42).randn(21))
        # A と B はほぼ同一→高相関
        returns = pd.DataFrame(
            {
                "A": np.diff(base),
                "B": np.diff(base) + np.random.RandomState(0).randn(20) * 0.01,
            }
        )
        result = _apply_correlation_penalty(symbols, weights, returns)
        # 重みが小さい方（A=0.5）が削減される
        self.assertLess(result["A"], 0.5)
        self.assertAlmostEqual(result["B"], 0.6)  # 大きい方は変わらない

    def test_empty_returns_returns_unchanged(self):
        symbols = ["A", "B"]
        weights = {"A": 0.4, "B": 0.6}
        result = _apply_correlation_penalty(symbols, weights, pd.DataFrame())
        self.assertDictEqual(result, weights)

    def test_single_symbol_returns_unchanged(self):
        symbols = ["A"]
        weights = {"A": 0.4}
        returns = pd.DataFrame({"A": np.random.randn(20)})
        result = _apply_correlation_penalty(symbols, weights, returns)
        self.assertAlmostEqual(result["A"], 0.4)


# ---------------------------------------------------------------------------
# _detect_regime
# ---------------------------------------------------------------------------


class TestDetectRegime(unittest.TestCase):
    def test_regime_override_returns_immediately(self):
        regime = _detect_regime("jp", "bear")
        self.assertEqual(regime, "bear")

    def test_regime_override_bull(self):
        self.assertEqual(_detect_regime("us", "bull"), "bull")

    def test_no_override_returns_unknown_on_empty_db(self):
        mock_con = MagicMock()
        mock_con.__enter__ = MagicMock(return_value=mock_con)
        mock_con.__exit__ = MagicMock(return_value=False)
        mock_con.execute.return_value.fetchone.return_value = None

        with patch("src.trading.capital_allocator._db_connection", return_value=mock_con):
            regime = _detect_regime("jp", None)
        self.assertEqual(regime, "unknown")


# ---------------------------------------------------------------------------
# _load_confidence_map
# ---------------------------------------------------------------------------


class TestLoadConfidenceMap(unittest.TestCase):
    def test_returns_empty_for_no_symbols(self):
        result = _load_confidence_map("jp", [])
        self.assertEqual(result, {})

    def test_returns_dir_accuracy_from_db(self):
        mock_con = MagicMock()
        mock_con.__enter__ = MagicMock(return_value=mock_con)
        mock_con.__exit__ = MagicMock(return_value=False)
        mock_con.execute.return_value.fetchall.return_value = [
            ("7203", 0.65, 20),
            ("9984", 0.72, 15),
        ]

        with patch("src.trading.capital_allocator._db_connection", return_value=mock_con):
            result = _load_confidence_map("jp", ["7203", "9984"])

        self.assertAlmostEqual(result["7203"], 0.65)
        self.assertAlmostEqual(result["9984"], 0.72)

    def test_db_error_returns_empty(self):
        mock_con = MagicMock()
        mock_con.__enter__ = MagicMock(return_value=mock_con)
        mock_con.__exit__ = MagicMock(return_value=False)
        mock_con.execute.side_effect = RuntimeError("DB error")

        with patch("src.trading.capital_allocator._db_connection", return_value=mock_con):
            result = _load_confidence_map("jp", ["7203"])
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# compute_portfolio_allocation
# ---------------------------------------------------------------------------


class TestComputePortfolioAllocation(unittest.TestCase):
    def setUp(self):
        # 共通パッチ
        self.patch_confidence = patch(
            "src.trading.capital_allocator._load_confidence_map",
            return_value={"7203": 0.65, "9984": 0.70, "6758": 0.60},
        )
        self.patch_returns = patch(
            "src.trading.capital_allocator._load_returns_for_corr",
            return_value=pd.DataFrame(),
        )
        self.patch_regime = patch(
            "src.trading.capital_allocator._detect_regime",
            return_value="bull",
        )
        self.patch_sector = patch(
            "src.trading.capital_allocator.get_symbol_sector",
            return_value=None,
        )
        self.patch_confidence.start()
        self.patch_returns.start()
        self.patch_regime.start()
        self.patch_sector.start()

    def tearDown(self):
        self.patch_confidence.stop()
        self.patch_returns.stop()
        self.patch_regime.stop()
        self.patch_sector.stop()

    def test_empty_candidates_returns_empty(self):
        result = compute_portfolio_allocation([], total_capital=1_000_000)
        self.assertEqual(result, [])

    def test_all_negative_signals_returns_empty(self):
        candidates = _make_candidates(["7203", "9984"], [-0.01, -0.02])
        result = compute_portfolio_allocation(candidates, total_capital=1_000_000)
        self.assertEqual(result, [])

    def test_returns_list_of_allocation_result(self):
        candidates = _make_candidates(["7203", "9984", "6758"], [0.02, 0.015, 0.01])
        results = compute_portfolio_allocation(candidates, total_capital=1_000_000)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, AllocationResult)

    def test_weight_cap_at_max_position_rate(self):
        from config.settings import MAX_POSITION_RATE

        # 1銘柄のみ → キャップに当たるはず
        candidates = _make_candidates(["7203"], [0.05])
        results = compute_portfolio_allocation(candidates, total_capital=1_000_000)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(results[0].weight, MAX_POSITION_RATE + 1e-9)

    def test_sorted_by_weight_descending(self):
        candidates = _make_candidates(["7203", "9984", "6758"], [0.03, 0.015, 0.005])
        results = compute_portfolio_allocation(candidates, total_capital=1_000_000)
        weights = [r.weight for r in results]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_regime_bear_reduces_allocation(self):
        with patch("src.trading.capital_allocator._detect_regime", return_value="bear"):
            candidates = _make_candidates(["7203", "9984"], [0.02, 0.015])
            bear_results = compute_portfolio_allocation(candidates, total_capital=1_000_000)

        with patch("src.trading.capital_allocator._detect_regime", return_value="bull"):
            candidates = _make_candidates(["7203", "9984"], [0.02, 0.015])
            bull_results = compute_portfolio_allocation(candidates, total_capital=1_000_000)

        # bear レジームの総配分は bull より小さいはず
        bear_total = sum(r.weight for r in bear_results)
        bull_total = sum(r.weight for r in bull_results)
        self.assertLessEqual(bear_total, bull_total)

    def test_allocated_capital_matches_weight_times_total(self):
        candidates = _make_candidates(["7203", "9984"], [0.02, 0.015])
        results = compute_portfolio_allocation(candidates, total_capital=2_000_000)
        for r in results:
            self.assertAlmostEqual(r.allocated_capital, r.weight * 2_000_000, places=2)

    def test_allocation_reason_contains_signal_info(self):
        candidates = _make_candidates(["7203"], [0.02])
        results = compute_portfolio_allocation(candidates, total_capital=1_000_000)
        self.assertGreater(len(results), 0)
        self.assertIn("signal=", results[0].allocation_reason)
        self.assertIn("confidence=", results[0].allocation_reason)

    def test_regime_stored_in_result(self):
        candidates = _make_candidates(["7203", "9984"], [0.02, 0.01])
        with patch("src.trading.capital_allocator._detect_regime", return_value="range"):
            results = compute_portfolio_allocation(
                candidates, total_capital=1_000_000, regime_override="range"
            )
        for r in results:
            self.assertEqual(r.regime, "range")

    def test_max_positions_limit(self):
        from config.settings import MAX_POSITIONS

        # MAX_POSITIONS より多い候補を渡す
        symbols = [str(i) for i in range(MAX_POSITIONS + 5)]
        diff_ratios = [0.01 + 0.001 * i for i in range(MAX_POSITIONS + 5)]
        candidates = _make_candidates(symbols, diff_ratios)

        with patch(
            "src.trading.capital_allocator._load_confidence_map",
            return_value={sym: 0.6 for sym in symbols},
        ):
            results = compute_portfolio_allocation(candidates, total_capital=1_000_000)
        self.assertLessEqual(len(results), MAX_POSITIONS)

    def test_position_size_ratio_is_weight_over_max_rate(self):
        from config.settings import MAX_POSITION_RATE

        candidates = _make_candidates(["7203", "9984"], [0.02, 0.01])
        results = compute_portfolio_allocation(candidates, total_capital=1_000_000)
        for r in results:
            expected_ratio = r.weight / MAX_POSITION_RATE if MAX_POSITION_RATE > 0 else 0.0
            self.assertAlmostEqual(r.position_size_ratio, expected_ratio, places=6)


if __name__ == "__main__":
    unittest.main()
