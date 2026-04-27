"""Discord のテキスト制限向けユーティリティ。"""

from __future__ import annotations

DISCORD_TEXT_LIMIT = 1900
DISCORD_WIDE_TEXT_LIMIT = 3800


def split_text_chunks(
    text: str, limit: int = DISCORD_TEXT_LIMIT, preserve_lines: bool = True
) -> list[str]:
    """Discord 送信用に文字列を分割する。"""
    if len(text) <= limit:
        return [text]
    if not preserve_lines:
        return [text[index : index + limit] for index in range(0, len(text), limit)]

    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(line) <= limit:
            current = line
            continue
        long_line_chunks = split_text_chunks(line, limit=limit, preserve_lines=False)
        chunks.extend(long_line_chunks[:-1])
        current = long_line_chunks[-1]
    if current:
        chunks.append(current)
    return chunks
