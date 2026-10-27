"""
ユニットテスト: correlation_risk

DuckDB は MagicMock で差し替える。
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.trading.correlation_risk import (
    CorrelationDataUnavailable,
    _load_recent_returns,
    compute_enc,
    evaluate_correlation_gate,
    filter_correlated_candidates,
)
from src.trading.types import CorrelationGateResult


def _run_load(frame: pd.DataFrame, symbols: list, window: int) -> pd.DataFrame:
    conn_cm = MagicMock()
    conn_cm.__enter__.return_value = MagicMock()
    with (
        patch("src.trading.correlation_risk._db_connection", return_value=conn_cm),
        patch("src.trading.correlation_risk.pd.read_sql", return_value=frame),
    ):
        return _load_recent_returns(symbols, "jp", window)


def _short_history_frame(short_bars: int) -> pd.DataFrame:
    """A と B は 21 本の日足（高い相関）、C は直近 short_bars 本だけの表。"""
    np.random.seed(7)
    dates = pd.bdate_range("2026-09-21", periods=21)
    base = 100.0 + np.cumsum(np.random.randn(21))
    closes = {
        "A": base,
        "B": base + np.random.randn(21) * 0.05,
        "C": 50.0 + np.cumsum(np.random.randn(21)),
    }
    rows = []
    for sym, values in closes.items():
        start = 21 - short_bars if sym == "C" else 0
        for ts, close in zip(dates[start:], values[start:]):
            rows.append({"symbol": sym, "ts": ts, "close": float(close)})
    return pd.DataFrame(rows)


def _stale_symbol_frame(stale_days: int) -> pd.DataFrame:
    """A と B は直近 21 本の日足（高い相関）、C は stale_days 営業日前で止まった 21 本の表。"""
    np.random.seed(11)
    dates = pd.bdate_range(end="2026-10-23", periods=21 + stale_days)
    base = 100.0 + np.cumsum(np.random.randn(21))
    closes = {
        "A": (dates[stale_days:], base),
        "B": (dates[stale_days:], base + np.random.randn(21) * 0.05),
        "C": (dates[:21], 50.0 + np.cumsum(np.random.randn(21))),
    }
    rows = []
    for sym, (sym_dates, values) in closes.items():
        for ts, close in zip(sym_dates, values):
            rows.append({"symbol": sym, "ts": ts, "close": float(close)})
    return pd.DataFrame(rows)


def _gap_symbol_frame(gap_days: int) -> pd.DataFrame:
    """A と B は直近 21 本の日足（高い相関）、C は直近から2日目以降の gap_days 日が欠けた 21 本の表。"""
    np.random.seed(13)
    dates = pd.bdate_range(end="2026-10-23", periods=21 + gap_days)
    base = 100.0 + np.cumsum(np.random.randn(21))
    c_dates = dates[:20].append(dates[-1:])
    closes = {
        "A": (dates[gap_days:], base),
        "B": (dates[gap_days:], base + np.random.randn(21) * 0.05),
        "C": (c_dates, 50.0 + np.cumsum(np.random.randn(21))),
    }
    rows = []
    for sym, (sym_dates, values) in closes.items():
        for ts, close in zip(sym_dates, values):
            rows.append({"symbol": sym, "ts": ts, "close": float(close)})
    return pd.DataFrame(rows)


class TestLoadRecentReturns(unittest.TestCase):
    """終値は stock_features ではなく market_data_raw の close から読む。

    stock_features はラグ特徴量（Close_lag1 等）しか持たず、引用符なしの
    close は本番の Postgres で "column does not exist" になっていた。
    """

    def _run(self, frame: pd.DataFrame):
        captured: dict = {}

        def fake_read_sql(query, con, params=None):
            captured["query"] = query
            captured["params"] = params
            return frame

        conn_cm = MagicMock()
        conn_cm.__enter__.return_value = MagicMock()
        with (
            patch("src.trading.correlation_risk._db_connection", return_value=conn_cm),
            patch("src.trading.correlation_risk.pd.read_sql", side_effect=fake_read_sql),
        ):
            result = _load_recent_returns(["A", "B"], "us", 2)
        return result, captured

    def test_reads_close_from_market_data_raw_daily_bars(self):
        frame = pd.DataFrame(
            {
                "symbol": ["A", "A", "A", "B", "B", "B"],
                "ts": pd.to_datetime(["2026-10-13", "2026-10-14", "2026-10-15"] * 2),
                "close": [100.0, 110.0, 99.0, 50.0, 55.0, 60.5],
            }
        )
        returns, captured = self._run(frame)

        query = " ".join(captured["query"].split())
        self.assertIn("FROM market_data_raw", query)
        self.assertNotIn("stock_features", query)
        self.assertIn("timeframe = %s", query)
        self.assertIn("PARTITION BY symbol ORDER BY ts DESC", query)
        self.assertEqual(captured["params"], ["us", "1d", "A", "B", 3])

        self.assertEqual(list(returns.columns), ["A", "B"])
        self.assertEqual(len(returns), 2)
        self.assertAlmostEqual(returns["A"].iloc[0], 0.10)
        self.assertAlmostEqual(returns["A"].iloc[1], -0.10)
        self.assertAlmostEqual(returns["B"].iloc[1], 0.10)

    def test_short_history_symbol_is_dropped_not_all_rows(self):
        """日足が1本の銘柄の列を外し、ほかの銘柄の行を残す（#1001）"""
        frame = _short_history_frame(short_bars=1)
        with self.assertLogs("src.trading.correlation_risk", level="WARNING") as logs:
            returns = _run_load(frame, ["A", "B", "C"], 20)

        self.assertEqual(list(returns.columns), ["A", "B"])
        self.assertEqual(len(returns), 20)
        self.assertTrue(any("C" in line for line in logs.output))

    def test_few_bars_symbol_is_dropped_and_rest_keep_window(self):
        """日足が数本の銘柄を外し、残りの銘柄の20日分を使う（#1001）"""
        frame = _short_history_frame(short_bars=4)
        returns = _run_load(frame, ["A", "B", "C"], 20)

        self.assertEqual(list(returns.columns), ["A", "B"])
        self.assertEqual(len(returns), 20)

    def test_stale_symbol_before_window_is_dropped_not_all_rows(self):
        """日足が25営業日前で止まった銘柄の列を外し、表を空にしない（#1141）"""
        frame = _stale_symbol_frame(stale_days=25)
        with self.assertLogs("src.trading.correlation_risk", level="WARNING") as logs:
            returns = _run_load(frame, ["A", "B", "C"], 20)

        self.assertEqual(list(returns.columns), ["A", "B"])
        self.assertEqual(len(returns), 20)
        self.assertTrue(any("'C'" in line for line in logs.output))

    def test_stale_symbol_inside_window_is_dropped_and_rest_keep_window(self):
        """日足が18営業日前で止まった銘柄を外し、残りの20日分を使う（#1141）"""
        frame = _stale_symbol_frame(stale_days=18)
        with self.assertLogs("src.trading.correlation_risk", level="WARNING") as logs:
            returns = _run_load(frame, ["A", "B", "C"], 20)

        self.assertEqual(list(returns.columns), ["A", "B"])
        self.assertEqual(len(returns), 20)
        self.assertTrue(any("'C'" in line for line in logs.output))

    def test_gap_symbol_inside_window_is_dropped_and_rest_keep_window(self):
        """日足が窓の途中で15営業日欠けた銘柄を外し、残りの20日分を使う（#1254）"""
        frame = _gap_symbol_frame(gap_days=15)
        with self.assertLogs("src.trading.correlation_risk", level="WARNING") as logs:
            returns = _run_load(frame, ["A", "B", "C"], 20)

        self.assertEqual(list(returns.columns), ["A", "B"])
        self.assertEqual(len(returns), 20)
        self.assertTrue(any("'C'" in line for line in logs.output))

    def test_two_symbols_with_one_common_row_returns_no_correlation_rows(self):
        """2銘柄で共通のリターンが1本なら、数本の相関の表を返さない（#1254）"""
        frame = _gap_symbol_frame(gap_days=18)
        returns = _run_load(frame[frame["symbol"] != "B"], ["A", "C"], 20)

        self.assertLess(len(returns.columns), 2)

    def test_all_symbols_short_returns_empty(self):
        frame = _short_history_frame(short_bars=1)
        returns = _run_load(frame[frame["symbol"] == "C"], ["C"], 20)
        self.assertTrue(returns.empty)

    def test_query_failure_raises(self):
        """取得の例外は空の表にせず投げる。空はデータ不足と区別がつかず素通しになる（#809）"""
        conn_cm = MagicMock()
        conn_cm.__enter__.side_effect = RuntimeError("boom")
        with patch("src.trading.correlation_risk._db_connection", return_value=conn_cm):
            with self.assertRaises(CorrelationDataUnavailable):
                _load_recent_returns(["A", "B"], "us", 2)


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
    def test_load_failure_blocks_new_entries(self, mock_load):
        """取得に失敗したら、新規の買いを止める側に倒す（#809）"""
        mock_load.side_effect = CorrelationDataUnavailable("boom")
        result = evaluate_correlation_gate(["A", "B"], "jp", window=20, enc_threshold=2.0)
        self.assertFalse(result.is_allowed)
        self.assertIn("boom", result.reason)

    def _gate_with_frame(self, frame: pd.DataFrame) -> CorrelationGateResult:
        conn_cm = MagicMock()
        conn_cm.__enter__.return_value = MagicMock()
        with (
            patch("src.trading.correlation_risk._db_connection", return_value=conn_cm),
            patch("src.trading.correlation_risk.pd.read_sql", return_value=frame),
        ):
            return evaluate_correlation_gate(["A", "B", "C"], "jp", window=20, enc_threshold=2.0)

    def test_one_bar_symbol_does_not_skip_gate(self):
        """3銘柄のうち C の日足が1本でも素通しにせず、A と B で ENC を出す（#1001）"""
        result = self._gate_with_frame(_short_history_frame(short_bars=1))

        self.assertNotEqual(result.reason, "リターンデータ不足のためスキップ")
        self.assertFalse(result.is_allowed)
        self.assertGreater(result.avg_correlation, 0.9)
        self.assertLess(result.enc, 1.1)

    def test_few_bars_symbol_excluded_and_rest_judged(self):
        """C の日足が数本なら C を外し、A と B の20日分の相関で判定する（#1001）"""
        frame = _short_history_frame(short_bars=4)
        expected = (
            frame[frame["symbol"] != "C"]
            .pivot(index="ts", columns="symbol", values="close")
            .pct_change()
            .dropna()
        )
        expected_corr = abs(expected["A"].corr(expected["B"]))

        result = self._gate_with_frame(frame)

        self.assertAlmostEqual(result.avg_correlation, expected_corr, places=10)
        self.assertAlmostEqual(result.enc, 2.0 / (1.0 + expected_corr), places=10)
        self.assertFalse(result.is_allowed)

    def _expected_ab_corr(self, frame: pd.DataFrame) -> float:
        expected = (
            frame[frame["symbol"] != "C"]
            .pivot(index="ts", columns="symbol", values="close")
            .pct_change()
            .dropna()
        )
        self.assertEqual(len(expected), 20)
        return float(abs(expected["A"].corr(expected["B"])))

    def test_stale_symbol_before_window_does_not_skip_gate(self):
        """C の日足が25営業日前で止まっても素通しにせず、A と B の20日分で ENC を出す（#1141）"""
        frame = _stale_symbol_frame(stale_days=25)
        expected_corr = self._expected_ab_corr(frame)

        result = self._gate_with_frame(frame)

        self.assertNotEqual(result.reason, "リターンデータ不足のためスキップ")
        self.assertAlmostEqual(result.avg_correlation, expected_corr, places=10)
        self.assertAlmostEqual(result.enc, 2.0 / (1.0 + expected_corr), places=10)
        self.assertFalse(result.is_allowed)

    def test_stale_symbol_inside_window_excluded_and_rest_judged(self):
        """C の日足が18営業日前で止まったら C を外し、A と B の20日分で判定する（#1141）"""
        frame = _stale_symbol_frame(stale_days=18)
        expected_corr = self._expected_ab_corr(frame)

        result = self._gate_with_frame(frame)

        self.assertAlmostEqual(result.avg_correlation, expected_corr, places=10)
        self.assertAlmostEqual(result.enc, 2.0 / (1.0 + expected_corr), places=10)
        self.assertFalse(result.is_allowed)

    def test_gap_symbol_inside_window_excluded_and_rest_judged(self):
        """C の日足が窓の途中で15営業日欠けたら C を外し、A と B の20日分で判定する（#1254）"""
        frame = _gap_symbol_frame(gap_days=15)
        expected_corr = self._expected_ab_corr(frame)

        result = self._gate_with_frame(frame)

        self.assertAlmostEqual(result.avg_correlation, expected_corr, places=10)
        self.assertAlmostEqual(result.enc, 2.0 / (1.0 + expected_corr), places=10)
        self.assertFalse(result.is_allowed)

    def test_one_common_row_does_not_judge_with_nan_enc(self):
        """共通のリターンが1本になる欠けでも、ENC を NaN のまま判定しない（#1254）"""
        frame = _gap_symbol_frame(gap_days=18)
        expected_corr = self._expected_ab_corr(frame)

        result = self._gate_with_frame(frame)

        self.assertFalse(np.isnan(result.enc))
        self.assertAlmostEqual(result.enc, 2.0 / (1.0 + expected_corr), places=10)

    def test_two_symbols_one_common_row_does_not_judge_with_nan_enc(self):
        """2銘柄で共通のリターンが1本なら、ENC=nan で止めずにデータ不足として扱う（#1254）"""
        frame = _gap_symbol_frame(gap_days=18)
        conn_cm = MagicMock()
        conn_cm.__enter__.return_value = MagicMock()
        with (
            patch("src.trading.correlation_risk._db_connection", return_value=conn_cm),
            patch(
                "src.trading.correlation_risk.pd.read_sql",
                return_value=frame[frame["symbol"] != "B"],
            ),
        ):
            result = evaluate_correlation_gate(["A", "C"], "jp", window=20, enc_threshold=2.0)

        self.assertFalse(np.isnan(result.enc))
        self.assertNotIn("nan", result.reason or "")

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
    def test_load_failure_blocks_all_candidates(self, mock_load):
        """取得に失敗したら、候補を全部止める（#809）"""
        mock_load.side_effect = CorrelationDataUnavailable("boom")
        allowed, blocked = filter_correlated_candidates(["A", "B"], ["EXIST"], "us")
        self.assertEqual(allowed, [])
        self.assertEqual(blocked, ["A", "B"])

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
