"""Anthropic SDK 経由の TextReviewPort 実装（API 従量課金）。

`anthropic.Anthropic()` を遅延 import で生成し、ANTHROPIC_API_KEY で認証する。
従来 reporting/llm_review.py・backtest/critical_review.py が直接行っていた
SDK 呼び出しを集約したもの。
"""

from typing import Any, Optional

from src.domain.ports import TextReviewPort
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnthropicSdkAdapter(TextReviewPort):
    """anthropic SDK をラップする TextReviewPort 実装。"""

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        schema: Optional[dict] = None,
    ) -> str:
        import anthropic  # noqa: PLC0415  遅延 import（未インストール環境を許容）

        client = anthropic.Anthropic()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        response = client.messages.create(**kwargs)
        return "".join(block.text for block in response.content if block.type == "text")
