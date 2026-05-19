"""
ユニットテスト: correlation_risk

DuckDB は MagicMock で差し替える。
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.trading.correlation_risk import (
    compute_enc,
    evaluate_correlation_gate,
    filter_correlated_candidates,
)
from src.trading.types import CorrelationGateResult


class TestComputeEnc(unittest.TestCase):
    def test_single_symbol_returns_one(self):
        corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
        enc, avg_corr = compute_enc(corr)
        self.assertAlmostEqual(enc, 1.0)
        self.assertAlmostEqual(avg_corr, 0.0)

    def test_zero_correlation_returns_n(self):
        # 完全無相関: ENC = N
        corr = pd.DataFrame(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        enc, avg_corr = compute_enc(corr)
        self.assertAlmostEqual(enc, 3.0)
        self.assertAlmostEqual(avg_corr, 0.0)

    def test_full_correlation_returns_one(self):
        # 完全相関: ENC = 1
        corr = pd.DataFrame(
            [[1.0, 1.0], [1.0, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        enc, avg_corr = compute_enc(corr)
        self.assertAlmostEqual(enc, 1.0)
        self.assertAlmostEqual(avg_corr, 1.0)

    def test_partial_correlation(self):
        # avg_corr=0.5 の 3 銘柄: ENC = 3/(1+0.5*2) = 1.5
        corr = pd.DataFrame(
            [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        enc, avg_corr = compute_enc(corr)
        self.assertAlmostEqual(enc, 1.5, places=5)
        self.assertAlmostEqual(avg_corr, 0.5, places=5)

    def test_negative_correlation_uses_abs(self):
        # 負の相関: abs を使うので avg_corr = 0.6
        corr = pd.DataFrame(
            [[1.0, -0.6], [-0.6, 1.0]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        enc, avg_corr = compute_enc(corr)
        self.assertAlmostEqual(avg_corr, 0.6, places=5)
        self.assertAlmostEqual(enc, 2.0 / (1.0 + 0.6 * 1), places=5)

    def test_empty_matrix_returns_zero(self):
        corr = pd.DataFrame()
        enc, avg_corr = compute_enc(corr)
        self.assertAlmostEqual(enc, 0.0)
        self.assertAlmostEqual(avg_corr, 0.0)


class TestEvaluateCorrelationGate(unittest.TestCase):
    def test_single_symbol_always_allowed(self):
        result = evaluate_correlation_gate(["7203"], "jp", window=20, enc_threshold=2.0)
        self.assertTrue(result.is_allowed)
        self.assertEqual(result.n_symbols, 1)

    def test_empty_symbols_always_allowed(self):
        result = evaluate_correlation_gate([], "jp", window=20, enc_threshold=2.0)
        self.assertTrue(result.is_allowed)

    @patch("src.trading.correlation_risk._load_recent_returns")
    def test_high_correlation_blocks_entry(self, mock_load):
        # avg_corr=0.9 の 3 銘柄: ENC ≈ 1.22 < 2.0 → ブロック
        np.random.seed(42)
        base = np.cumsum(np.random.randn(25))
        noise_b = base + np.random.randn(25) * 0.05
        noise_c = base + np.random.randn(25) * 0.05
        df = pd.DataFrame({"A": np.diff(base), "B": np.diff(noise_b), "C": np.diff(noise_c)})
        mock_load.return_value = df

        result = evaluate_correlation_gate(["A", "B", "C"], "jp", window=20, enc_threshold=2.0)
        self.assertFalse(result.is_allowed)
        self.assertIsNotNone(result.reason)
        self.assertLess(result.enc, 2.0)

    @patch("src.trading.correlation_risk._load_recent_returns")
    def test_low_correlation_allows_entry(self, mock_load):
        # 独立したリターン → ENC ≈ N
        np.random.seed(0)
        df = pd.DataFrame(
            {
                "A": np.random.randn(20),
                "B": np.random.randn(20),
                "C": np.random.randn(20),
            }
        )
        mock_load.return_value = df

        result = evaluate_correlation_gate(["A", "B", "C"], "jp", window=20, enc_threshold=2.0)
        self.assertTrue(result.is_allowed)
        self.assertGreaterEqual(result.enc, 2.0)
        self.assertIsNone(result.reason)

    @patch("src.trading.correlation_risk._load_recent_returns")
    def test_empty_returns_skips_gate(self, mock_load):
        mock_load.return_value = pd.DataFrame()
        result = evaluate_correlation_gate(["A", "B"], "jp", window=20, enc_threshold=2.0)
        self.assertTrue(result.is_allowed)
        self.assertIsNotNone(result.reason)

    @patch("src.trading.correlation_risk._load_recent_returns")
    def test_returns_correct_type(self, mock_load):
        np.random.seed(1)
        df = pd.DataFrame({"X": np.random.randn(20), "Y": np.random.randn(20)})
        mock_load.return_value = df

        result = evaluate_correlation_gate(["X", "Y"], "jp")
        self.assertIsInstance(result, CorrelationGateResult)
        self.assertIsInstance(result.enc, float)
        self.assertIsInstance(result.avg_correlation, float)


class TestFilterCorrelatedCandidates(unittest.TestCase):
    def test_no_existing_positions_allows_all(self):
        allowed, blocked = filter_correlated_candidates(["A", "B"], [], "us")
        self.assertEqual(allowed, ["A", "B"])
        self.assertEqual(blocked, [])

    def test_no_candidates_returns_empty(self):
        allowed, blocked = filter_correlated_candidates([], ["X"], "us")
        self.assertEqual(allowed, [])
        self.assertEqual(blocked, [])

    @patch("src.trading.correlation_risk._load_recent_returns")
    def test_high_corr_candidate_blocked(self, mock_load):
        np.random.seed(0)
        base = np.cumsum(np.random.randn(65))
        noise = base + np.random.randn(65) * 0.03
        df = pd.DataFrame({"NEW": np.diff(base), "EXIST": np.diff(noise)})
        mock_load.return_value = df

        allowed, blocked = filter_correlated_candidates(
            ["NEW"], ["EXIST"], "us", window=60, threshold=0.7
        )
        self.assertEqual(blocked, ["NEW"])
        self.assertEqual(allowed, [])

    @patch("src.trading.correlation_risk._load_recent_returns")
    def test_low_corr_candidate_allowed(self, mock_load):
        np.random.seed(42)
        df = pd.DataFrame({"NEW": np.random.randn(60), "EXIST": np.random.randn(60)})
        mock_load.return_value = df

        allowed, blocked = filter_correlated_candidates(
            ["NEW"], ["EXIST"], "us", window=60, threshold=0.7
        )
        self.assertEqual(allowed, ["NEW"])
        self.assertEqual(blocked, [])

    @patch("src.trading.correlation_risk._load_recent_returns")
    def test_empty_returns_allows_all(self, mock_load):
        mock_load.return_value = pd.DataFrame()

        allowed, blocked = filter_correlated_candidates(
            ["A", "B"], ["C"], "us", window=60, threshold=0.7
        )
        self.assertEqual(allowed, ["A", "B"])
        self.assertEqual(blocked, [])

    @patch("src.trading.correlation_risk._load_recent_returns")
    def test_candidate_not_in_data_allowed(self, mock_load):
        df = pd.DataFrame({"EXIST": np.random.randn(60)})
        mock_load.return_value = df

        allowed, blocked = filter_correlated_candidates(
            ["MISSING"], ["EXIST"], "us", window=60, threshold=0.7
        )
        self.assertEqual(allowed, ["MISSING"])
        self.assertEqual(blocked, [])

    @patch("src.trading.correlation_risk._load_recent_returns")
    def test_mixed_candidates(self, mock_load):
        np.random.seed(1)
        base = np.cumsum(np.random.randn(65))
        noise = base + np.random.randn(65) * 0.02
        independent = np.random.randn(64)
        df = pd.DataFrame(
            {"HIGH_CORR": np.diff(base), "LOW_CORR": independent, "EXIST": np.diff(noise)}
        )
        mock_load.return_value = df

        allowed, blocked = filter_correlated_candidates(
            ["HIGH_CORR", "LOW_CORR"], ["EXIST"], "us", window=60, threshold=0.7
        )
        self.assertIn("HIGH_CORR", blocked)
        self.assertIn("LOW_CORR", allowed)


if __name__ == "__main__":
    unittest.main()
