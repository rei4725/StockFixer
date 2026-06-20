"""LLM_BACKEND に応じて TextReviewPort 実装を選択する factory。"""

from config.settings import CLAUDE_CLI_BINARY, CLAUDE_CLI_TIMEOUT_SECONDS, LLM_BACKEND
from src.domain.ports import TextReviewPort
from src.infrastructure.llm.cli_adapter import ClaudeCliAdapter
from src.infrastructure.llm.sdk_adapter import AnthropicSdkAdapter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_text_review_port() -> TextReviewPort:
    """設定 LLM_BACKEND に従って TextReviewPort を返す。

    "cli" → Claude Code CLI（サブスク認証）、それ以外 → Anthropic SDK（API 課金）。
    """
    backend = (LLM_BACKEND or "sdk").strip().lower()
    if backend == "cli":
        logger.debug("[llm_factory] CLI バックエンドを選択（サブスク認証）")
        return ClaudeCliAdapter(
            binary=CLAUDE_CLI_BINARY, timeout_seconds=CLAUDE_CLI_TIMEOUT_SECONDS
        )
    if backend != "sdk":
        logger.warning("[llm_factory] 未知の LLM_BACKEND=%s。sdk にフォールバック", backend)
    return AnthropicSdkAdapter()
