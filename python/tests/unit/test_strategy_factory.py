"""戦略ファクトリー（#369 Phase 1）のユニットテスト"""

import json
import math
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.backtest.factory import (
    _window_bounds,
    apply_gate,
    build_rule,
    control_hypotheses,
    evaluate_hypothesis,
    run_factory_batch,
    sample_hypotheses,
    write_report,
)
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

_ATOMIC_SPEC = {
    "type": "atomic",
    "rule": "ema_momentum",
    "params": {"fast_window": 8, "slow_window": 21},
}


def _make_ohlcv(n=600, seed=42, trend=0.0005):
    """合成 OHLCV（上昇トレンド + ノイズ）を生成する。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(trend, 0.015, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, n)),
            "High": close * (1 + abs(rng.normal(0, 0.006, n))),
            "Low": close * (1 - abs(rng.normal(0, 0.006, n))),
            "Close": close,
            "Volume": rng.integers(1000, 50000, n).astype(float),
        },
        index=dates,
    )


class TestFactoryHypothesis(unittest.TestCase):
    """FactoryHypothesis のハッシュ・ラベルのテスト"""

    def test_hash_is_deterministic(self):
        h1 = FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp")
        h2 = FactoryHypothesis(rule_spec=dict(_ATOMIC_SPEC), market="jp")
        self.assertEqual(h1.hypothesis_hash, h2.hypothesis_hash)
        self.assertEqual(len(h1.hypothesis_hash), 12)

    def test_hash_differs_by_spec_and_market(self):
        h1 = FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp")
        h2 = FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="us")
        h3 = FactoryHypothesis(
            rule_spec={"type": "atomic", "rule": "rsi_contrarian", "params": {}},
            market="jp",
        )
        self.assertNotEqual(h1.hypothesis_hash, h2.hypothesis_hash)
        self.assertNotEqual(h1.hypothesis_hash, h3.hypothesis_hash)

    def test_is_control_does_not_change_hash(self):
        h1 = FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp", is_control=False)
        h2 = FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp", is_control=True)
        self.assertEqual(h1.hypothesis_hash, h2.hypothesis_hash)

    def test_label_for_composite(self):
        spec = {
            "type": "and",
            "rules": [
                _ATOMIC_SPEC,
                {"type": "atomic", "rule": "volume_breakout", "params": {"volume_ratio": 2.0}},
            ],
        }
        h = FactoryHypothesis(rule_spec=spec, market="jp")
        self.assertIn("and(", h.label)
        self.assertIn("ema_momentum", h.label)
        self.assertIn("volume_breakout", h.label)


class TestBuildRule(unittest.TestCase):
    """build_rule のテスト"""

    def test_atomic_rule_with_params(self):
        rule = build_rule(_ATOMIC_SPEC)
        self.assertEqual(rule.name, "ema_momentum")
        self.assertEqual(rule.fast_window, 8)
        self.assertEqual(rule.slow_window, 21)

    def test_and_composite(self):
        spec = {
            "type": "and",
            "rules": [
                _ATOMIC_SPEC,
                {"type": "atomic", "rule": "rsi_contrarian", "params": {}},
            ],
        }
        rule = build_rule(spec)
        self.assertTrue(rule.name.startswith("and_"))

    def test_unknown_rule_raises(self):
        with self.assertRaises(ValueError):
            build_rule({"type": "atomic", "rule": "nonexistent", "params": {}})

    def test_composite_with_one_child_raises(self):
        with self.assertRaises(ValueError):
            build_rule({"type": "and", "rules": [_ATOMIC_SPEC]})


class TestSampler(unittest.TestCase):
    """sample_hypotheses / control_hypotheses のテスト"""

    def test_respects_budget(self):
        sampled = sample_hypotheses("jp", budget=5, existing_hashes=set(), seed=1)
        self.assertEqual(len(sampled), 5)

    def test_no_duplicates_within_batch_or_existing(self):
        first = sample_hypotheses("jp", budget=8, existing_hashes=set(), seed=1)
        hashes = {h.hypothesis_hash for h in first}
        self.assertEqual(len(hashes), 8)
        second = sample_hypotheses("jp", budget=8, existing_hashes=hashes, seed=1)
        self.assertTrue(hashes.isdisjoint({h.hypothesis_hash for h in second}))

    def test_deterministic_with_seed(self):
        a = sample_hypotheses("jp", budget=5, existing_hashes=set(), seed=7)
        b = sample_hypotheses("jp", budget=5, existing_hashes=set(), seed=7)
        self.assertEqual([h.hypothesis_hash for h in a], [h.hypothesis_hash for h in b])

    def test_controls_are_six_default_atomics(self):
        controls = control_hypotheses("jp")
        self.assertEqual(len(controls), 6)
        self.assertTrue(all(c.is_control for c in controls))
        self.assertTrue(all(c.rule_spec["type"] == "atomic" for c in controls))


class TestApplyGate(unittest.TestCase):
    """apply_gate のテスト"""

    def _make_eval(self, **kwargs):
        defaults = dict(
            hypothesis=FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp"),
            sharpe_ratio=2.0,
            num_trades=50,
            max_drawdown=-0.10,
            dsr=0.97,
            pbo=0.30,
        )
        defaults.update(kwargs)
        return FactoryEvaluation(**defaults)

    def test_passes_when_all_conditions_met(self):
        ev = self._make_eval()
        apply_gate(ev, champion_sharpe=1.0)
        self.assertTrue(ev.gate_passed)
        self.assertEqual(ev.gate_reasons, [])

    def test_fails_on_low_trades(self):
        ev = self._make_eval(num_trades=10)
        apply_gate(ev, champion_sharpe=1.0)
        self.assertFalse(ev.gate_passed)
        self.assertTrue(any("num_trades" in r for r in ev.gate_reasons))

    def test_fails_on_low_dsr(self):
        ev = self._make_eval(dsr=0.80)
        apply_gate(ev, champion_sharpe=1.0)
        self.assertFalse(ev.gate_passed)

    def test_high_pbo_does_not_block_gate(self):
        # PBO はバッチ診断へ降格。高 PBO でも他条件を満たせばゲートは通る。
        ev = self._make_eval(pbo=0.70)
        apply_gate(ev, champion_sharpe=1.0)
        self.assertTrue(ev.gate_passed)
        self.assertFalse(any("pbo" in r for r in ev.gate_reasons))

    def test_fails_on_deep_drawdown(self):
        ev = self._make_eval(max_drawdown=-0.40)
        apply_gate(ev, champion_sharpe=1.0)
        self.assertFalse(ev.gate_passed)

    def test_fails_when_not_beating_champion(self):
        # champion マージン=1.0: champion 以下なら不合格
        ev = self._make_eval(sharpe_ratio=0.9)
        apply_gate(ev, champion_sharpe=1.0)  # 必要値 1.0
        self.assertFalse(ev.gate_passed)
        self.assertTrue(any("sharpe" in r for r in ev.gate_reasons))

    def test_champion_nan_skips_champion_condition(self):
        ev = self._make_eval(sharpe_ratio=0.5, dsr=0.97, pbo=0.3)
        apply_gate(ev, champion_sharpe=float("nan"))
        self.assertTrue(ev.gate_passed)


class TestEvaluateHypothesis(unittest.TestCase):
    """evaluate_hypothesis のテスト（合成データ・実 Backtester）"""

    def test_returns_window_returns_of_expected_length(self):
        data = {"AAA": _make_ohlcv(seed=1), "BBB": _make_ohlcv(seed=2)}
        windows = _window_bounds("2024-01-01", "2026-04-01", 8)
        h = FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp")

        ev = evaluate_hypothesis(h, data, windows)

        self.assertEqual(len(ev.window_returns), 8)
        self.assertEqual(ev.n_symbols, 2)
        self.assertIsInstance(ev.sharpe_ratio, float)

    def test_zero_signal_rule_yields_empty_metrics(self):
        # rsi_contrarian は rsi 列がないとシグナルゼロ → 取引なし
        data = {"AAA": _make_ohlcv(seed=1)}
        windows = _window_bounds("2024-01-01", "2026-04-01", 4)
        h = FactoryHypothesis(
            rule_spec={"type": "atomic", "rule": "rsi_contrarian", "params": {}},
            market="jp",
        )

        ev = evaluate_hypothesis(h, data, windows)

        self.assertEqual(ev.num_trades, 0)
        self.assertEqual(ev.window_returns, [0.0] * 4)


class TestWriteReport(unittest.TestCase):
    """write_report のテスト"""

    def test_writes_immutable_json_with_contract_fields(self):
        ev = FactoryEvaluation(
            hypothesis=FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp"),
            sharpe_ratio=1.5,
            dsr=0.96,
            pbo=0.2,
            num_trades=40,
            max_drawdown=-0.1,
            window_returns=[0.01] * 8,
            n_symbols=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory.get_results_dir", return_value=tmp):
                path = write_report(ev, champion_sharpe=1.0, period=("2024-01-01", "2026-01-01"))

            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".tmp"))
            with open(path, encoding="utf-8") as f:
                report = json.load(f)

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["hypothesis_hash"], ev.hypothesis.hypothesis_hash)
        self.assertIn(f"[factory:{ev.hypothesis.hypothesis_hash}]", report["issue_title"])
        self.assertIn("strategy-factory", report["labels"])
        self.assertIn("Deflated Sharpe", report["issue_body"])
        self.assertEqual(report["gate"]["dsr"], 0.96)
        # 低 PBO のときは警告注記を出さない
        self.assertNotIn("バッチPBO", report["issue_body"])

    def test_high_batch_pbo_adds_warning_to_body(self):
        ev = FactoryEvaluation(
            hypothesis=FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp"),
            sharpe_ratio=1.5,
            dsr=0.96,
            pbo=0.70,  # 高 PBO
            num_trades=40,
            max_drawdown=-0.1,
            window_returns=[0.01] * 8,
            n_symbols=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory.get_results_dir", return_value=tmp):
                path = write_report(ev, champion_sharpe=1.0, period=("2024-01-01", "2026-01-01"))
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        self.assertIn("バッチPBO", report["issue_body"])
        self.assertIn("過学習リスク", report["issue_body"])


class TestRunFactoryBatch(unittest.TestCase):
    """run_factory_batch の結合テスト（ポート・DB をモック）"""

    def _fake_port(self):
        port = MagicMock()
        port.download.side_effect = lambda ticker, start=None, end=None: _make_ohlcv(
            seed=hash(ticker) % 100
        )
        port.add_technical_indicators.side_effect = lambda df: df
        return port

    @patch("src.backtest.factory.save_factory_run")
    @patch("src.backtest.factory.count_factory_runs", return_value=0)
    @patch("src.backtest.factory.load_factory_hashes", return_value=set())
    @patch("src.backtest.factory.get_backtest_data_port")
    def test_batch_evaluates_and_records_candidates(
        self, mock_port, mock_hashes, mock_count, mock_save
    ):
        mock_port.return_value = self._fake_port()

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory.get_results_dir", return_value=tmp):
                result = run_factory_batch(
                    market="jp", symbols=["AAA", "BBB"], budget=4, n_windows=6, seed=123
                )

                self.assertEqual(len(result.candidates), 4)
                # 候補は全件 DB 記録される
                self.assertEqual(mock_save.call_count, 4)
                # DSR / PBO が全候補に付与される
                for ev in result.candidates:
                    self.assertFalse(math.isnan(ev.dsr))
                # 合格仮説にはレポートが書かれている
                for ev in result.passed:
                    self.assertIsNotNone(ev.report_path)
                    self.assertTrue(os.path.exists(ev.report_path))

    @patch("src.backtest.factory.save_factory_run")
    @patch("src.backtest.factory.count_factory_runs", return_value=0)
    @patch("src.backtest.factory.load_factory_hashes", return_value=set())
    @patch("src.backtest.factory.get_backtest_data_port")
    def test_batch_aborts_without_symbol_data(self, mock_port, mock_hashes, mock_count, mock_save):
        port = MagicMock()
        port.download.return_value = None
        mock_port.return_value = port

        result = run_factory_batch(market="jp", symbols=["AAA"], budget=3, seed=1)

        self.assertEqual(result.evaluated, [])
        mock_save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
