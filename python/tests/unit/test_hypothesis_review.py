"""ユニットテスト: 仮説単位の批判的レビュー（src/backtest/hypothesis_review.py）

anthropic クライアントは MagicMock で差し替え、API 呼び出しは行わない。
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.backtest import hypothesis_review
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

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


def _mock_anthropic(review: dict):
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps(review)
    response = MagicMock()
    response.content = [text_block]
    client = MagicMock()
    client.messages.create.return_value = response
    module = MagicMock()
    module.Anthropic.return_value = client
    return module, client


_SAMPLE_REVIEW = {
    "risk_level": "medium",
    "assessment": "窓7のみリターンが突出しており、他窓は横ばい。",
    "concerns": ["窓7への依存度が高い", "取引数に対しSharpeがやや高い"],
}


# 環境変数 LLM_BACKEND に依存させず、anthropic モックが効く SDK 経路に固定する
@patch("src.infrastructure.llm.factory.LLM_BACKEND", "sdk")
class TestReviewHypothesis(unittest.TestCase):
    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", False)
    def test_disabled_returns_none(self):
        ev = _make_evaluation()
        self.assertIsNone(hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0))

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_enabled_returns_parsed_review(self):
        ev = _make_evaluation()
        module, client = _mock_anthropic(_SAMPLE_REVIEW)
        with patch.dict("sys.modules", {"anthropic": module}):
            result = hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0)
        self.assertEqual(result, _SAMPLE_REVIEW)
        # 構造化出力を要求していること
        _, kwargs = client.messages.create.call_args
        self.assertIn("output_config", kwargs)

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_api_error_returns_none(self):
        ev = _make_evaluation()
        module = MagicMock()
        module.Anthropic.side_effect = RuntimeError("API down")
        with patch.dict("sys.modules", {"anthropic": module}):
            self.assertIsNone(hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0))

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_malformed_schema_returns_none(self):
        ev = _make_evaluation()
        module, _client = _mock_anthropic({"unexpected": "shape"})
        with patch.dict("sys.modules", {"anthropic": module}):
            self.assertIsNone(hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0))

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_champion_nan_does_not_raise(self):
        ev = _make_evaluation()
        module, _client = _mock_anthropic(_SAMPLE_REVIEW)
        with patch.dict("sys.modules", {"anthropic": module}):
            result = hypothesis_review.review_hypothesis(ev, champion_sharpe=float("nan"))
        self.assertEqual(result, _SAMPLE_REVIEW)


class TestBuildReviewContext(unittest.TestCase):
    def test_context_includes_spec_and_metrics(self):
        ev = _make_evaluation()
        context = hypothesis_review._build_review_context(ev, champion_sharpe=1.0)
        self.assertIn("ema_momentum", context)
        self.assertIn("DSR", context.replace(" ", "").replace("(", "").replace(")", "") + "DSR")
        self.assertIn("0.970", context)  # dsr
        self.assertIn("窓1", context)


if __name__ == "__main__":
    unittest.main()
