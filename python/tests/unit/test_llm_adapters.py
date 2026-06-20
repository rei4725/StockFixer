"""ユニットテスト: LLM レビュー用アダプタと factory（src/infrastructure/llm/）

subprocess・anthropic SDK は MagicMock で差し替え、外部呼び出しは行わない。
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.infrastructure.llm.cli_adapter import ClaudeCliAdapter, _strip_fences
from src.infrastructure.llm.factory import get_text_review_port
from src.infrastructure.llm.sdk_adapter import AnthropicSdkAdapter


class TestStripFences(unittest.TestCase):
    def test_removes_json_fence(self):
        self.assertEqual(_strip_fences('```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_removes_bare_fence(self):
        self.assertEqual(_strip_fences("```\nhello\n```"), "hello")

    def test_plain_text_untouched(self):
        self.assertEqual(_strip_fences("  plain text  "), "plain text")


def _fake_proc(payload: dict) -> MagicMock:
    proc = MagicMock()
    proc.stdout = json.dumps(payload)
    return proc


class TestClaudeCliAdapter(unittest.TestCase):
    def test_complete_parses_result_and_builds_command(self):
        adapter = ClaudeCliAdapter(binary="claude", timeout_seconds=5)
        with patch(
            "src.infrastructure.llm.cli_adapter.subprocess.run",
            return_value=_fake_proc({"is_error": False, "result": "答え"}),
        ) as mock_run:
            out = adapter.complete(system="sys", user="usr", model="haiku", max_tokens=100)

        self.assertEqual(out, "答え")
        cmd = mock_run.call_args[0][0]
        # 決定論化フラグが付いていること
        self.assertEqual(cmd[0], "claude")
        self.assertIn("-p", cmd)
        self.assertIn("--tools", cmd)
        self.assertIn("--setting-sources", cmd)
        self.assertIn("--output-format", cmd)

    def test_complete_strips_fences_from_result(self):
        adapter = ClaudeCliAdapter()
        with patch(
            "src.infrastructure.llm.cli_adapter.subprocess.run",
            return_value=_fake_proc({"is_error": False, "result": '```json\n{"x":1}\n```'}),
        ):
            out = adapter.complete(system="s", user="u", model="m", max_tokens=10)
        self.assertEqual(out, '{"x":1}')

    def test_schema_appends_instruction_to_prompt(self):
        adapter = ClaudeCliAdapter()
        with patch(
            "src.infrastructure.llm.cli_adapter.subprocess.run",
            return_value=_fake_proc({"is_error": False, "result": "{}"}),
        ) as mock_run:
            adapter.complete(
                system="s", user="u", model="m", max_tokens=10, schema={"type": "object"}
            )
        prompt = mock_run.call_args[0][0][2]  # cmd[2] = -p の本文
        self.assertIn("JSON Schema", prompt)
        self.assertIn('"type"', prompt)

    def test_is_error_raises(self):
        adapter = ClaudeCliAdapter()
        with patch(
            "src.infrastructure.llm.cli_adapter.subprocess.run",
            return_value=_fake_proc({"is_error": True, "result": "Not logged in"}),
        ):
            with self.assertRaises(RuntimeError):
                adapter.complete(system="s", user="u", model="m", max_tokens=10)


def _mock_anthropic(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create.return_value = response
    module = MagicMock()
    module.Anthropic.return_value = client
    return module, client


class TestAnthropicSdkAdapter(unittest.TestCase):
    def test_complete_joins_text_and_passes_model(self):
        module, client = _mock_anthropic("こんにちは")
        with patch.dict("sys.modules", {"anthropic": module}):
            out = AnthropicSdkAdapter().complete(
                system="s", user="u", model="claude-x", max_tokens=50
            )
        self.assertEqual(out, "こんにちは")
        _, kwargs = client.messages.create.call_args
        self.assertEqual(kwargs["model"], "claude-x")
        self.assertNotIn("output_config", kwargs)

    def test_complete_with_schema_passes_output_config(self):
        module, client = _mock_anthropic('{"findings": []}')
        with patch.dict("sys.modules", {"anthropic": module}):
            AnthropicSdkAdapter().complete(
                system="s", user="u", model="m", max_tokens=5, schema={"type": "object"}
            )
        _, kwargs = client.messages.create.call_args
        self.assertIn("output_config", kwargs)


class TestFactory(unittest.TestCase):
    @patch("src.infrastructure.llm.factory.LLM_BACKEND", "cli")
    def test_cli_backend_returns_cli_adapter(self):
        self.assertIsInstance(get_text_review_port(), ClaudeCliAdapter)

    @patch("src.infrastructure.llm.factory.LLM_BACKEND", "sdk")
    def test_sdk_backend_returns_sdk_adapter(self):
        self.assertIsInstance(get_text_review_port(), AnthropicSdkAdapter)

    @patch("src.infrastructure.llm.factory.LLM_BACKEND", "unknown")
    def test_unknown_backend_falls_back_to_sdk(self):
        self.assertIsInstance(get_text_review_port(), AnthropicSdkAdapter)


if __name__ == "__main__":
    unittest.main()
