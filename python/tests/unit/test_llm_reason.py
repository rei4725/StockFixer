import unittest
from typing import Optional
from unittest.mock import MagicMock, patch

from src.domain.types import PredictionResult
from src.reporting.llm_reason import generate_reason, generate_reasons_for_top


def _make_result(
    symbol: str = "7203",
    market: str = "jp",
    diff_ratio: float = 0.02,
    confidence_ratio: Optional[float] = 0.8,
) -> PredictionResult:
    return PredictionResult(
        market=market,
        symbol=symbol,
        current_price=1000.0,
        avg_pred_price=1020.0,
        diff_ratio=diff_ratio,
        model_count=2,
        confidence_ratio=confidence_ratio,
    )


class TestGenerateReason(unittest.TestCase):
    @patch("src.utils.ollama_client.OllamaClient")
    def test_returns_none_when_ollama_unavailable(self, mock_cls):
        mock_client = MagicMock()
        mock_client.is_available = False
        mock_cls.return_value = mock_client

        result = generate_reason(_make_result())
        self.assertIsNone(result)

    @patch("src.utils.ollama_client.OllamaClient")
    def test_returns_reason_string_when_available(self, mock_cls):
        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.generate_text.return_value = "業績改善期待から買い優勢となっている。"
        mock_cls.return_value = mock_client

        result = generate_reason(_make_result(diff_ratio=0.03))

        self.assertIsNotNone(result)
        self.assertIn("業績", result)
        mock_client.generate_text.assert_called_once()

    @patch("src.utils.ollama_client.OllamaClient")
    def test_returns_none_when_generate_text_returns_none(self, mock_cls):
        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.generate_text.return_value = None
        mock_cls.return_value = mock_client

        result = generate_reason(_make_result())
        self.assertIsNone(result)

    @patch("src.utils.ollama_client.OllamaClient")
    def test_confidence_none_does_not_raise(self, mock_cls):
        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.generate_text.return_value = "理由文。"
        mock_cls.return_value = mock_client

        result = generate_reason(_make_result(confidence_ratio=None))
        self.assertIsNotNone(result)

    @patch("src.utils.ollama_client.OllamaClient")
    def test_negative_diff_ratio_uses_down_direction(self, mock_cls):
        mock_client = MagicMock()
        mock_client.is_available = True
        mock_client.generate_text.return_value = "売り圧力が強い。"
        mock_cls.return_value = mock_client

        generate_reason(_make_result(diff_ratio=-0.05))

        call_prompt = mock_client.generate_text.call_args[0][0]
        self.assertIn("下落予測", call_prompt)


class TestGenerateReasonsForTop(unittest.TestCase):
    @patch("src.utils.ollama_client.OllamaClient")
    def test_returns_empty_dict_when_unavailable(self, mock_cls):
        mock_client = MagicMock()
        mock_client.is_available = False
        mock_cls.return_value = mock_client

        results = [_make_result(symbol=s) for s in ("7203", "6758", "9984")]
        reasons = generate_reasons_for_top(results)
        self.assertEqual(reasons, {})

    @patch("src.reporting.llm_reason.generate_reason")
    @patch("src.utils.ollama_client.OllamaClient")
    def test_caps_at_max_count(self, mock_cls, mock_gen):
        mock_client = MagicMock()
        mock_client.is_available = True
        mock_cls.return_value = mock_client
        mock_gen.return_value = "理由。"

        results = [_make_result(symbol=str(i)) for i in range(10)]
        generate_reasons_for_top(results, max_count=3)

        self.assertEqual(mock_gen.call_count, 3)

    @patch("src.reporting.llm_reason.generate_reason")
    @patch("src.utils.ollama_client.OllamaClient")
    def test_skips_none_reasons(self, mock_cls, mock_gen):
        mock_client = MagicMock()
        mock_client.is_available = True
        mock_cls.return_value = mock_client
        mock_gen.side_effect = ["理由A。", None, "理由C。"]

        results = [_make_result(symbol=s) for s in ("A", "B", "C")]
        reasons = generate_reasons_for_top(results, max_count=3)

        self.assertIn("A", reasons)
        self.assertNotIn("B", reasons)
        self.assertIn("C", reasons)


if __name__ == "__main__":
    unittest.main()
