"""LLM レビュー用アダプタ群（TextReviewPort の実装）。

- AnthropicSdkAdapter: ANTHROPIC_API_KEY による API 従量課金経路
- ClaudeCliAdapter:    Claude Code CLI（サブスク認証）経路
- get_text_review_port: LLM_BACKEND に応じて適切な実装を返す factory
"""

from src.infrastructure.llm.factory import get_text_review_port

__all__ = ["get_text_review_port"]
