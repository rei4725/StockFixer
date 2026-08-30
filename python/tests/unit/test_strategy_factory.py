"""戦略ファクトリー（#369 Phase 1）のユニットテスト"""

import json
import math
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.backtest.factory import (
    _MIN_SYMBOL_ROWS,
    _WARMUP_CALENDAR_DAYS,
    _load_symbol_data,
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
from src.market_data.technical import add_technical_indicators

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
            n_effective_symbols=50,
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


class TestLoadSymbolData(unittest.TestCase):
    """_load_symbol_data の助走（ウォームアップ）区間処理のテスト（#629）"""

    def _ohlcv(self, start, periods=400):
        dates = pd.bdate_range(start=start, periods=periods)
        close = 100 + np.arange(len(dates), dtype=float)
        return pd.DataFrame(
            {
                "Open": close,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": np.full(len(dates), 10000.0),
            },
            index=dates,
        )

    @patch("src.backtest.factory.get_backtest_data_port")
    def test_download_start_is_extended_by_warmup(self, mock_get_port):
        port = MagicMock()
        recorded = {}

        def _download(ticker, start=None, end=None):
            recorded["start"] = start
            recorded["end"] = end
            return self._ohlcv(start="2024-01-01")

        port.download.side_effect = _download
        port.add_technical_indicators.side_effect = lambda df: df
        mock_get_port.return_value = port

        _load_symbol_data("jp", ["AAA"], start="2024-06-01", end="2024-12-01")

        expected_warmup_start = (
            pd.Timestamp("2024-06-01") - pd.Timedelta(days=_WARMUP_CALENDAR_DAYS)
        ).strftime("%Y-%m-%d")
        self.assertEqual(recorded["start"], expected_warmup_start)
        self.assertEqual(recorded["end"], "2024-12-01")

    @patch("src.backtest.factory.get_backtest_data_port")
    def test_warmup_rows_are_trimmed_from_result(self, mock_get_port):
        """助走区間ぶんダウンロードしても、返却データはstart以降のみ。"""
        port = MagicMock()
        port.download.side_effect = lambda ticker, start=None, end=None: self._ohlcv(start=start)
        port.add_technical_indicators.side_effect = lambda df: df
        mock_get_port.return_value = port

        data = _load_symbol_data("jp", ["AAA"], start="2024-06-01", end="2024-12-01")

        self.assertIn("AAA", data)
        self.assertGreaterEqual(data["AAA"].index.min(), pd.Timestamp("2024-06-01"))

    @patch("src.backtest.factory.get_backtest_data_port")
    def test_thin_eval_window_is_skipped_even_with_enough_raw_rows(self, mock_get_port):
        """助走込みの生データが_MIN_SYMBOL_ROWS以上でも、評価期間側が薄ければ除外する。"""
        port = MagicMock()
        # 助走開始日から _MIN_SYMBOL_ROWS を超える営業日数を返すが、大半が助走区間に
        # 落ちるため、start以降（評価期間側）は _MIN_SYMBOL_ROWS を割り込む設定。
        raw_periods = _MIN_SYMBOL_ROWS + 30
        port.download.side_effect = lambda ticker, start=None, end=None: self._ohlcv(
            start=start, periods=raw_periods
        )
        port.add_technical_indicators.side_effect = lambda df: df
        mock_get_port.return_value = port

        data = _load_symbol_data("jp", ["AAA"], start="2024-06-01", end="2024-12-01")

        self.assertNotIn("AAA", data)


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
            with patch("src.backtest.factory_report.get_results_dir", return_value=tmp):
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
            with patch("src.backtest.factory_report.get_results_dir", return_value=tmp):
                path = write_report(ev, champion_sharpe=1.0, period=("2024-01-01", "2026-01-01"))
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        self.assertIn("バッチPBO", report["issue_body"])
        self.assertIn("過学習リスク", report["issue_body"])

    def test_review_section_embedded_when_provided(self):
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
        review = {
            "risk_level": "high",
            "assessment": "窓1のみに依存した成績。",
            "concerns": ["窓1以外はほぼ横ばい"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory_report.get_results_dir", return_value=tmp):
                path = write_report(
                    ev, champion_sharpe=1.0, period=("2024-01-01", "2026-01-01"), review=review
                )
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        self.assertIn("Claude批判的レビュー", report["issue_body"])
        self.assertIn("窓1のみに依存した成績。", report["issue_body"])
        self.assertIn("窓1以外はほぼ横ばい", report["issue_body"])
        self.assertIn("risk_level=high", report["issue_body"])
        self.assertEqual(report["review"], review)

    def test_review_section_omitted_when_none(self):
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
            with patch("src.backtest.factory_report.get_results_dir", return_value=tmp):
                path = write_report(ev, champion_sharpe=1.0, period=("2024-01-01", "2026-01-01"))
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        self.assertNotIn("Claude批判的レビュー", report["issue_body"])
        self.assertIsNone(report["review"])


class TestRunFactoryBatch(unittest.TestCase):
    """run_factory_batch の結合テスト（ポート・DB をモック）"""

    # #628: この銘柄・データパラメータ・budget/seed の組み合わせは、実際に合格仮説
    # （macd_rsi(rsi_filter=70.0): trades=56, dsr=0.970, dd=-0.183, sharpe=1.414,
    # champion=1.286）が1件出ることを事前に確認済み（決定論的・再現性確認済み）。
    # 単なる合成データではなく実際のシグナルが出るよう、port.add_technical_indicators
    # には本物の実装を使う（旧: 恒等関数だったため bb/macd/rsi/atr 依存のルールが
    # 一切シグナルを出せず、合格経路が実質テストされていなかった）。
    _PASSING_SYMBOLS = ["SYM_0", "SYM_1", "SYM_2"]
    _PASSING_BATCH_KWARGS = {"budget": 8, "n_windows": 6, "seed": 1}
    _PASSING_DATA_PERIODS = 700
    _PASSING_DATA_TREND = 0.002
    _PASSING_DATA_VOL = 0.015
    _PASSING_DATA_SEED_BASE = 1
    # #628当初は「今日」を終端とする営業日レンジで合成データを生成していたが、
    # run_factory_batch側もdatetime.now()でstart/end文字列を計算しており、
    # 暦日ベースのウィンドウ境界と営業日のみのデータ点が噛み合うかどうかが
    # 実行日の曜日次第でわずかに変動し、PBOがちょうど0.50のしきい値付近を
    # 跨いで合格件数が0になる（=このテストが落ちる）ことがあった
    # （2026-08-30(日)にローカル・CI双方で再現・確認済み）。
    # 「今日」への依存自体を無くし、以前実際に合格することを確認済みの日付
    # (2026-08-29、develop上のCIで合格実績あり)へ凍結することで解消する。
    _FROZEN_NOW = datetime(2026, 8, 29)

    def _fake_port(self):
        """本物の add_technical_indicators を使い、実際に合格候補が出る合成データを返す。

        データは _FROZEN_NOW を終端とする営業日レンジで生成する（実行日の
        カレンダー日付には一切依存しないため、いつ実行しても結果は同じになる）。
        """
        seeds = {
            symbol: self._PASSING_DATA_SEED_BASE + i
            for i, symbol in enumerate(self._PASSING_SYMBOLS)
        }
        frozen_now = self._FROZEN_NOW

        def _ohlcv_ending_today(seed):
            rng = np.random.default_rng(seed)
            end = pd.Timestamp(frozen_now).normalize()
            dates = pd.bdate_range(end=end, periods=self._PASSING_DATA_PERIODS)
            trend, vol = self._PASSING_DATA_TREND, self._PASSING_DATA_VOL
            close = 100 * np.exp(np.cumsum(rng.normal(trend, vol, len(dates))))
            return pd.DataFrame(
                {
                    "Open": close * (1 + rng.normal(0, 0.003, len(dates))),
                    "High": close * (1 + abs(rng.normal(0, vol / 2, len(dates)))),
                    "Low": close * (1 - abs(rng.normal(0, vol / 2, len(dates)))),
                    "Close": close,
                    "Volume": rng.integers(1000, 50000, len(dates)).astype(float),
                },
                index=dates,
            )

        def _download(ticker, start=None, end=None):
            for symbol, seed in seeds.items():
                if ticker.startswith(symbol):
                    return _ohlcv_ending_today(seed)
            raise AssertionError(f"想定外のticker: {ticker}")

        port = MagicMock()
        port.download.side_effect = _download
        port.add_technical_indicators.side_effect = add_technical_indicators
        return port

    @patch("src.backtest.factory.datetime")
    @patch("src.backtest.factory.FACTORY_CLAUDE_RULEGEN_ENABLED", False)
    @patch("src.backtest.factory.review_hypothesis", return_value=None)
    @patch("src.backtest.factory.FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS", 1)
    @patch("src.backtest.factory.save_factory_run")
    @patch("src.backtest.factory.count_factory_runs", return_value=0)
    @patch("src.backtest.factory.load_factory_hashes", return_value=set())
    @patch("src.backtest.factory.get_backtest_data_port")
    def test_batch_evaluates_and_records_candidates(
        self, mock_port, mock_hashes, mock_count, mock_save, mock_review, mock_datetime
    ):
        mock_datetime.now.return_value = self._FROZEN_NOW
        mock_port.return_value = self._fake_port()

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory_report.get_results_dir", return_value=tmp):
                result = run_factory_batch(
                    market="jp", symbols=self._PASSING_SYMBOLS, **self._PASSING_BATCH_KWARGS
                )

                self.assertEqual(len(result.candidates), self._PASSING_BATCH_KWARGS["budget"])
                # 候補は全件 DB 記録される
                self.assertEqual(mock_save.call_count, self._PASSING_BATCH_KWARGS["budget"])
                # DSR / PBO が全候補に付与される
                for ev in result.candidates:
                    self.assertFalse(math.isnan(ev.dsr))
                # 合格経路が実際に実行されていることを検証する（#628）
                self.assertGreater(len(result.passed), 0)
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

    @patch("src.backtest.factory.datetime")
    @patch("src.backtest.factory.FACTORY_CLAUDE_RULEGEN_ENABLED", False)
    @patch("src.backtest.factory.FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS", 1)
    @patch("src.backtest.factory.review_hypothesis")
    @patch("src.backtest.factory.save_factory_run")
    @patch("src.backtest.factory.count_factory_runs", return_value=0)
    @patch("src.backtest.factory.load_factory_hashes", return_value=set())
    @patch("src.backtest.factory.get_backtest_data_port")
    def test_batch_calls_review_only_for_passed_hypotheses(
        self, mock_port, mock_hashes, mock_count, mock_save, mock_review, mock_datetime
    ):
        mock_datetime.now.return_value = self._FROZEN_NOW
        mock_port.return_value = self._fake_port()
        mock_review.return_value = None

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory_report.get_results_dir", return_value=tmp):
                result = run_factory_batch(
                    market="jp", symbols=self._PASSING_SYMBOLS, **self._PASSING_BATCH_KWARGS
                )

        # 合格経路が実際に実行されていることを検証する（#628: result.passed が常に
        # 空だと call_count の 0 == 0 比較で自明に真になってしまっていた）
        self.assertGreater(len(result.passed), 0)
        self.assertEqual(mock_review.call_count, len(result.passed))

    @patch("src.backtest.factory.datetime")
    @patch("src.backtest.factory.FACTORY_CLAUDE_RULEGEN_ENABLED", False)
    @patch("src.backtest.factory.FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS", 1)
    @patch("src.backtest.factory.review_hypothesis", return_value=None)
    @patch("src.backtest.factory.save_factory_run")
    @patch("src.backtest.factory.count_factory_runs", return_value=0)
    @patch("src.backtest.factory.load_factory_hashes", return_value=set())
    @patch("src.backtest.factory.get_backtest_data_port")
    def test_review_none_still_writes_report(
        self, mock_port, mock_hashes, mock_count, mock_save, mock_review, mock_datetime
    ):
        # review_hypothesis はグレースフルデグラデーション契約により失敗時 None を返す
        # （例外を投げない）。None が返ってもレポート書き込みは通常通り完了する。
        mock_datetime.now.return_value = self._FROZEN_NOW
        mock_port.return_value = self._fake_port()

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory_report.get_results_dir", return_value=tmp):
                result = run_factory_batch(
                    market="jp", symbols=self._PASSING_SYMBOLS, **self._PASSING_BATCH_KWARGS
                )

            # 合格経路が実際に実行されていることを検証する（#628）。
            # tempdir が生きている間にファイル存在チェックまで行う（#628調査中に
            # 発見: 元のコードは tempdir を抜けた後に os.path.exists していたため、
            # result.passed が常に空だった旧実装ではこのバグも顕在化していなかった）。
            self.assertGreater(len(result.passed), 0)
            for evaluation in result.passed:
                self.assertIsNotNone(evaluation.report_path)
                self.assertTrue(os.path.exists(evaluation.report_path))
                with open(evaluation.report_path, encoding="utf-8") as f:
                    report = json.load(f)
                self.assertIsNone(report["review"])


if __name__ == "__main__":
    unittest.main()
