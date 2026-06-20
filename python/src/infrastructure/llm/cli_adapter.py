"""Claude Code CLI 経由の TextReviewPort 実装（サブスク認証）。

`claude -p` を subprocess 起動し、Pro/Max サブスク枠（または
CLAUDE_CODE_OAUTH_TOKEN）で LLM 出力を得る。API 従量課金が不要になる。

決定論化のためツールとプロジェクト設定を無効化する（--tools "" /
--setting-sources ""）。これを怠ると claude がリポジトリを勝手に読み、
出力に無関係な内容が混入する。
"""

import json
import re
import subprocess
from typing import Optional

from src.domain.ports import TextReviewPort
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 先頭/末尾のコードフェンス（```json ... ```）を除去するための正規表現。
# CLI は schema 準拠 JSON をフェンスで包んで返すことがある。
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_SCHEMA_INSTRUCTION = (
    "\n\n以下の JSON Schema に厳密準拠した JSON のみを出力せよ"
    "（コードフェンス・前置き・後書きは一切不要）:\n"
)


def _strip_fences(text: str) -> str:
    """応答テキストの先頭/末尾コードフェンスを除去する。"""
    return _FENCE_RE.sub("", text).strip()


class ClaudeCliAdapter(TextReviewPort):
    """`claude -p` を subprocess 起動する TextReviewPort 実装。"""

    def __init__(self, binary: str = "claude", timeout_seconds: int = 180) -> None:
        self._binary = binary
        self._timeout = timeout_seconds

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,  # CLI には対応する引数がないためベストエフォート（未使用）
        schema: Optional[dict] = None,
    ) -> str:
        prompt = user
        if schema is not None:
            prompt = user + _SCHEMA_INSTRUCTION + json.dumps(schema, ensure_ascii=False)

        cmd = [
            self._binary,
            "-p",
            prompt,
            "--append-system-prompt",
            system,
            "--model",
            model,
            "--output-format",
            "json",
            "--tools",
            "",  # ツールを無効化（ファイル/Bash を触らせない）
            "--setting-sources",
            "",  # プロジェクト/ユーザ設定・CLAUDE.md を読ませない
        ]

        proc = subprocess.run(  # nosec B603 B607
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self._timeout,
            check=True,
        )
        envelope = json.loads(proc.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude CLI returned error: {envelope.get('result')}")
        return _strip_fences(str(envelope.get("result", "")))
