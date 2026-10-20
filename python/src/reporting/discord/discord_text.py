"""Discord のテキスト制限向けユーティリティ。"""

from __future__ import annotations

DISCORD_TEXT_LIMIT = 1900
DISCORD_WIDE_TEXT_LIMIT = 3800

# Discord API の1通の上限（超えると 400 Bad Request になる。#810）
DISCORD_CONTENT_LIMIT = 2000
DISCORD_EMBED_TITLE_LIMIT = 256
DISCORD_EMBED_DESCRIPTION_LIMIT = 4096
DISCORD_EMBED_FIELDS_LIMIT = 25
DISCORD_EMBED_FIELD_NAME_LIMIT = 256
DISCORD_EMBED_FIELD_VALUE_LIMIT = 1024
DISCORD_EMBED_FOOTER_LIMIT = 2048
DISCORD_EMBED_TOTAL_LIMIT = 6000

# 日時表示フォーマット（Discord 通知共通）
DISCORD_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S JST"
DISCORD_MINUTE_FORMAT = "%Y-%m-%d %H:%M JST"
DISCORD_DATE_FORMAT = "%Y/%m/%d"


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


def truncate_text(text: str, limit: int) -> str:
    """limit 字を超える文を、末尾を「…」にして limit 字に切り詰める。"""
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    return text[: limit - 1] + "…"


def fit_lines(lines: list[str], limit: int = DISCORD_TEXT_LIMIT, unit: str = "行") -> str:
    """先頭から limit 字に収まるだけの行をつなぎ、省いた行の数を末尾に書く。

    Args:
        lines: つなぐ行（大事なものから順に並べておく）
        limit: つないだ文の字数の上限
        unit: 省いた数の単位（1行が1銘柄なら「銘柄」）
    """
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    reserve = len(f"\n…ほか {len(lines)} {unit}を省略")
    kept: list[str] = []
    used = 0
    for line in lines:
        added = len(line) + (1 if kept else 0)
        if used + added + reserve > limit:
            break
        kept.append(line)
        used += added
    note = f"…ほか {len(lines) - len(kept)} {unit}を省略"
    if not kept:
        return truncate_text(text, limit - len(note) - 1) + "\n" + note
    return "\n".join(kept) + "\n" + note
