"""ユニットテスト: 仮説単位の批判的レビュー（src/backtest/hypothesis_review.py）

LLM は InMemoryTextReviewPort を注入し、API 呼び出しは行わない。
"""

import json
import unittest
from unittest.mock import patch

from src.backtest import hypothesis_review
from src.backtest.types import FactoryEvaluation, FactoryHypothesis
from src.infrastructure.in_memory import InMemoryTextReviewPort

_SPEC = {"type": "atomic", "rule": "ema_momentum", "params": {"fast_window": 8, "slow_window": 21}}


def _make_evaluation(**kwargs):
    defaults = dict(
        hypothesis=FactoryHypothesis(rule_spec=_SPEC, market="jp"),
        sharpe_ratio=1.8,
        dsr=0.97,
        pbo=0.25,
        num_trades=45,
        max_drawdown=-0.12,
        win_rate=0.55,
        total_return=0.30,
        window_returns=[0.02, -0.01, 0.03, 0.01, 0.00, -0.02, 0.04, 0.01],
        n_symbols=5,
    )
    defaults.update(kwargs)
    return FactoryEvaluation(**defaults)


def _port(review: dict) -> InMemoryTextReviewPort:
    return InMemoryTextReviewPort([json.dumps(review)])


_SAMPLE_REVIEW = {
    "risk_level": "medium",
    "assessment": "窓7のみリターンが突出しており、他窓は横ばい。",
    "concerns": ["窓7への依存度が高い", "取引数に対しSharpeがやや高い"],
}


class TestReviewHypothesis(unittest.TestCase):
    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", False)
    def test_disabled_returns_none(self):
        ev = _make_evaluation()
        port = _port(_SAMPLE_REVIEW)
        self.assertIsNone(
            hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0, review_port=port)
        )
        self.assertEqual(port.calls, [])

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_enabled_returns_parsed_review(self):
        ev = _make_evaluation()
        port = _port(_SAMPLE_REVIEW)
        result = hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0, review_port=port)
        self.assertEqual(result, _SAMPLE_REVIEW)
        # 構造化出力を要求していること
        self.assertEqual(port.calls[0]["schema"], hypothesis_review._REVIEW_SCHEMA)

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_api_error_returns_none(self):
        ev = _make_evaluation()
        port = InMemoryTextReviewPort(error=RuntimeError("API down"))
        self.assertIsNone(
            hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0, review_port=port)
        )

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_malformed_schema_returns_none(self):
        ev = _make_evaluation()
        port = _port({"unexpected": "shape"})
        self.assertIsNone(
            hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0, review_port=port)
        )

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_champion_nan_does_not_raise(self):
        ev = _make_evaluation()
        result = hypothesis_review.review_hypothesis(
            ev, champion_sharpe=float("nan"), review_port=_port(_SAMPLE_REVIEW)
        )
        self.assertEqual(result, _SAMPLE_REVIEW)


class TestBuildReviewContext(unittest.TestCase):
    def test_context_includes_spec_and_metrics(self):
        ev = _make_evaluation()
        context = hypothesis_review._build_review_context(ev, champion_sharpe=1.0)
        self.assertIn("ema_momentum", context)
        self.assertIn("DSR", context.replace(" ", "").replace("(", "").replace(")", "") + "DSR")
        self.assertIn("0.970", context)  # dsr
        self.assertIn("窓1", context)

    def test_context_includes_symbol_denominators(self):
        """レビュアーが Sharpe の母数を誤認しないよう母数を渡す（#625）。"""
        evaluation = FactoryEvaluation(
            hypothesis=FactoryHypothesis(
                rule_spec={"type": "atomic", "rule": "rsi_contrarian", "params": {}},
                market="jp",
            ),
            sharpe_ratio=1.6,
            dsr=0.99,
            pbo=0.1,
            num_trades=85,
            max_drawdown=-0.19,
            win_rate=0.85,
            total_return=0.09,
            window_returns=[0.01],
            n_symbols=194,
            n_symbols_with_signal=69,
            n_effective_symbols=16,
            avg_trades_per_symbol=1.23,
        )

        context = hypothesis_review._build_review_context(evaluation, champion_sharpe=1.083)

        self.assertIn("データ取得銘柄数: 194", context)
        self.assertIn(
            "Sharpe（有効銘柄平均・プール値算出不能によりゲート判定に使用）: 1.600", context
        )
        self.assertIn(
            "最大DD（有効銘柄の最悪値・曲線欠損によりゲート判定に使用）: -19.00%", context
        )
        self.assertIn("取引数（有効銘柄合計）: 85", context)
        self.assertIn("シグナル発生銘柄数: 69", context)
        self.assertIn("有効銘柄数（集計母数）: 16", context)
        self.assertIn("銘柄あたり平均取引数（シグナル発生銘柄基準）: 1.23", context)
        self.assertIn("勝率（有効銘柄平均）: 85.00%", context)
        self.assertIn("リターン（有効銘柄平均）: 9.00%", context)

    def _pooled_evaluation(self, **overrides):
        values = dict(
            hypothesis=FactoryHypothesis(rule_spec=_SPEC, market="jp"),
            sharpe_ratio=0.573,
            portfolio_sharpe_ratio=8.841,
            sharpe_per_trade=0.263,
            dsr=1.0,
            pbo=0.543,
            num_trades=2253,
            max_drawdown=-0.6783,
            portfolio_max_drawdown=-0.0754,
            window_returns=[0.01],
            n_symbols=194,
        )
        values.update(overrides)
        return FactoryEvaluation(**values)

    def test_context_puts_gate_sharpe_next_to_champion(self):
        """チャンピオンと比べる Sharpe は、ゲートと同じプール済み年率値であること（#738）。

        銘柄別平均（0.573）しか渡さないと、プール値のチャンピオン（7.199）と
        単位の違う数字を比べて「桁違いに低い」と誤読される。
        """
        context = hypothesis_review._build_review_context(
            self._pooled_evaluation(), champion_sharpe=7.199
        )

        self.assertIn("Sharpe（プール済み取引リターンを年率化・ゲート判定用）: 8.841", context)
        self.assertIn("対照群（チャンピオン）Sharpe（ゲート判定用と同じ指標）: 7.199", context)
        self.assertIn("1取引あたり Sharpe（プール済み）: 0.263", context)
        self.assertIn("Sharpe（有効銘柄平均・診断用）: 0.573", context)

    def test_context_puts_portfolio_drawdown_before_worst_symbol(self):
        """ゲート対象のポートフォリオDDを渡し、最悪銘柄DDは診断値と明示する（#738）。"""
        context = hypothesis_review._build_review_context(
            self._pooled_evaluation(), champion_sharpe=7.199
        )

        self.assertIn(
            "最大DD（有効銘柄を等金額保有したポートフォリオ・ゲート判定用）: -7.54%", context
        )
        self.assertIn("最大DD（有効銘柄の最悪値・診断用）: -67.83%", context)

    def test_context_marks_pbo_as_batch_diagnostic(self):
        """PBO はバッチ単位の診断値でゲートに使っていないことを明示する。"""
        context = hypothesis_review._build_review_context(
            self._pooled_evaluation(), champion_sharpe=7.199
        )

        self.assertIn("PBO（バッチ全体の診断値・ゲート判定には不使用）: 0.543", context)

    def test_context_falls_back_to_mean_sharpe_when_pooled_is_nan(self):
        """プール値が算出不能ならゲートと同じく銘柄平均を判定用として渡す。"""
        context = hypothesis_review._build_review_context(
            self._pooled_evaluation(
                portfolio_sharpe_ratio=float("nan"), portfolio_max_drawdown=float("nan")
            ),
            champion_sharpe=7.199,
        )

        self.assertIn(
            "Sharpe（有効銘柄平均・プール値算出不能によりゲート判定に使用）: 0.573", context
        )
        self.assertIn(
            "最大DD（有効銘柄の最悪値・曲線欠損によりゲート判定に使用）: -67.83%", context
        )
        self.assertNotIn("ポートフォリオ", context)


if __name__ == "__main__":
    unittest.main()
