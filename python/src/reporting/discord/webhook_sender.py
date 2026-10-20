"""
Discord Webhook 送信基盤

低レベルの webhook トランスポート層。URL 取得・POST・チャンク分割・
ステータス通知・ファイル送信など、ドメイン非依存の送信プリミティブを提供する。
discord_utils.py のドメイン通知関数はここを土台に使う（#497 段階的分割の第1弾）。
"""

import logging
import os
from typing import Optional

import requests

from src.reporting.discord import rate_limiter as _rate_limiter
from src.reporting.discord.discord_notification_specs import NotificationSpec
from src.reporting.discord.discord_text import (
    DISCORD_CONTENT_LIMIT,
    DISCORD_EMBED_DESCRIPTION_LIMIT,
    DISCORD_EMBED_FIELD_NAME_LIMIT,
    DISCORD_EMBED_FIELD_VALUE_LIMIT,
    DISCORD_EMBED_FIELDS_LIMIT,
    DISCORD_EMBED_FOOTER_LIMIT,
    DISCORD_EMBED_TITLE_LIMIT,
    DISCORD_EMBED_TOTAL_LIMIT,
    DISCORD_TEXT_LIMIT,
    fit_lines,
    split_text_chunks,
    truncate_text,
)
from src.utils.japan_time import isoformat_jst
from src.utils.run_context import get_run_id

logger = logging.getLogger(__name__)

_ERROR_BODY_LIMIT = 1000


def _describe_error(exc: requests.exceptions.RequestException) -> str:
    """送信の失敗を、webhook の URL を含めずに、状態コードと応答の本文で表す（#810）。"""
    response = getattr(exc, "response", None)
    if response is None:
        return type(exc).__name__
    body = response.text if isinstance(response.text, str) else ""
    return f"HTTP {response.status_code} 応答={body[:_ERROR_BODY_LIMIT]}"


def _fit_embed(embed: dict) -> dict:
    """embed を Discord の上限（title・description・fields・合計の字数）の内に収める。"""
    embed["title"] = truncate_text(embed.get("title", ""), DISCORD_EMBED_TITLE_LIMIT)
    fields = embed.get("fields")
    if fields:
        embed["fields"] = [
            {
                **field,
                "name": truncate_text(str(field.get("name", "")), DISCORD_EMBED_FIELD_NAME_LIMIT),
                "value": truncate_text(
                    str(field.get("value", "")), DISCORD_EMBED_FIELD_VALUE_LIMIT
                ),
            }
            for field in fields[:DISCORD_EMBED_FIELDS_LIMIT]
        ]
    footer = embed.get("footer")
    if footer:
        footer["text"] = truncate_text(footer.get("text", ""), DISCORD_EMBED_FOOTER_LIMIT)

    others = len(embed["title"]) + len(footer["text"] if footer else "")
    for field in embed.get("fields", []):
        others += len(field["name"]) + len(field["value"])
    # fields だけで合計の上限を超えるときは、後ろの field から落とす
    while embed.get("fields") and others > DISCORD_EMBED_TOTAL_LIMIT:
        dropped = embed["fields"].pop()
        others -= len(dropped["name"]) + len(dropped["value"])
    description_limit = min(DISCORD_EMBED_DESCRIPTION_LIMIT, DISCORD_EMBED_TOTAL_LIMIT - others)
    embed["description"] = fit_lines(
        str(embed.get("description", "")).split("\n"), limit=description_limit
    )
    return embed


def _get_webhook_url() -> str | None:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.warning(
            "DISCORD_WEBHOOK_URLが環境変数に設定されていません。Webhook通知をスキップします。"
        )
        return None
    return webhook_url


def _post_webhook(
    *,
    json_payload: dict | None = None,
    data_payload: dict | None = None,
    files=None,
    timeout: int = 10,
):
    webhook_url = _get_webhook_url()
    if not webhook_url:
        return None
    _rate_limiter.apply_rate_limit()
    response = requests.post(
        webhook_url, json=json_payload, data=data_payload, files=files, timeout=timeout
    )
    response.raise_for_status()
    return response


def send_webhook_text_chunked(
    text: str,
    *,
    limit: int = DISCORD_TEXT_LIMIT,
    preserve_lines: bool = True,
) -> bool:
    success = True
    for chunk in split_text_chunks(text, limit=limit, preserve_lines=preserve_lines):
        if not send_webhook_text(chunk):
            success = False
    return success


def send_text_file_chunked(
    file_path: str,
    *,
    limit: int = DISCORD_TEXT_LIMIT,
    preserve_lines: bool = False,
) -> bool:
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            return send_webhook_text_chunked(
                file_handle.read(),
                limit=limit,
                preserve_lines=preserve_lines,
            )
    except OSError as exc:
        logger.error("テキストファイル送信失敗: %s", exc, exc_info=True)
        return False


def send_webhook_notification(
    title: str,
    message: str,
    color: int = 0x00FF00,
    fields: Optional[list[dict]] = None,
) -> bool:
    """
    Discordブhookを使用して通知を送信する

    Args:
        title: メッセージタイトル
        message: メッセージ本文
        color: Embedの色（16進数、デフォルトは緑）

    Returns:
        成功時True、失敗時False
    """
    try:
        should_send, suppression_summary = _rate_limiter.check_and_record(f"{title}\n{message}")
        if not should_send:
            return True

        if suppression_summary:
            try:
                _post_webhook(json_payload={"content": suppression_summary}, timeout=10)
            except requests.exceptions.RequestException as e:
                logger.error("抑止サマリー送信失敗: %s", _describe_error(e), exc_info=True)

        run_id = get_run_id()
        embed: dict = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": isoformat_jst(),
        }
        if fields:
            embed["fields"] = fields
        if run_id:
            embed["footer"] = {"text": f"run_id: {run_id}"}
        embed_data = {"embeds": [_fit_embed(embed)]}

        response = _post_webhook(json_payload=embed_data, timeout=10)
        if response is None:
            return False
        response.raise_for_status()

        logger.info("Discord通知送信成功: %s", title)
        return True

    except requests.exceptions.RequestException as e:
        logger.error("Discord通知送信失敗: %s: %s", title, _describe_error(e), exc_info=True)
        return False


def send_webhook_text(text: str) -> bool:
    """
    プレーンテキストメッセージをWebhookで送信する

    Discord の1通の上限（DISCORD_CONTENT_LIMIT 字）を超える文は、行を保って複数の通に分ける。

    Args:
        text: 送信テキスト（コードフェンス込み）

    Returns:
        成功時True、失敗時False
    """
    chunks = [text] if len(text) <= DISCORD_CONTENT_LIMIT else split_text_chunks(text)
    try:
        for chunk in chunks:
            response = _post_webhook(json_payload={"content": chunk}, timeout=10)
            if response is None:
                return False
            response.raise_for_status()

        logger.info("Discord通知送信成功: テキストメッセージ")
        return True

    except requests.exceptions.RequestException as e:
        logger.error("Discord通知送信失敗: %s", _describe_error(e), exc_info=True)
        return False


def send_webhook_file(file_path: str, title: str = "") -> bool:
    """
    ファイル（PNG等）を Discord Webhook にアップロードして送信する。

    Args:
        file_path: 送信するファイルの絶対パス
        title: 添付メッセージ（省略可）

    Returns:
        成功時True、失敗時False
    """
    if not os.path.exists(file_path):
        logger.warning("送信対象ファイルが存在しません: %s", file_path)
        return False

    try:
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            payload = {"content": title} if title else {}
            response = _post_webhook(
                data_payload=payload,
                files={"file": (filename, f)},
                timeout=30,
            )
        if response is None:
            return False
        response.raise_for_status()
        logger.info("Discordファイル送信成功: %s", filename)
        return True
    except requests.exceptions.RequestException as e:
        logger.error("Discordファイル送信失敗: %s", _describe_error(e))
        return False


def send_status_notification(spec: NotificationSpec, lines: list[str]) -> bool:
    return send_webhook_notification(spec.title, "\n".join(lines), color=spec.color)


def send_status_fields(
    spec: NotificationSpec,
    fields: list[dict],
    description: str = "",
) -> bool:
    """NotificationSpec と embed fields（名前/値の2カラムグリッド）で通知する。

    Args:
        spec: タイトルと色
        fields: [{"name": str, "value": str, "inline": bool}, ...]
        description: fields の上に表示する補足文（任意）
    """
    return send_webhook_notification(spec.title, description, color=spec.color, fields=fields)
