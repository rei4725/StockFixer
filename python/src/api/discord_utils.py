"""
Discord通知ユーティリティ

Webhookを使用したDiscord通知機能
"""

import logging
import os
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def send_webhook_notification(
    title: str,
    message: str,
    color: int = 0x00FF00,
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
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        logger.warning(
            "DISCORD_WEBHOOK_URLが環境変数に設定されていません。" "Webhook通知をスキップします。"
        )
        return False

    try:
        embed_data = {
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": datetime.now().isoformat(),
                }
            ]
        }

        response = requests.post(webhook_url, json=embed_data, timeout=10)
        response.raise_for_status()

        logger.info(f"Discord通知送信成功: {title}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Discord通知送信失敗: {e}")
        return False


def send_webhook_text(text: str) -> bool:
    """
    プレーンテキストメッセージをWebhookで送信する

    Args:
        text: 送信テキスト（コードフェンス込み）

    Returns:
        成功時True、失敗時False
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        logger.warning(
            "DISCORD_WEBHOOK_URLが環境変数に設定されていません。" "Webhook通知をスキップします。"
        )
        return False

    try:
        payload = {"content": text}
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        logger.info("Discord通知送信成功: テキストメッセージ")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Discord通知送信失敗: {e}")
        return False


def send_daily_pipeline_completion(
    data_count: Optional[int] = None,
    prediction_markets: Optional[list] = None,
    include_forecast: bool = True,
) -> bool:
    """
    日次パイプライン完了通知（完了メッセージ + 予測結果テーブル）

    Args:
        data_count: 取得したデータ件数
        prediction_markets: 予測対象マーケット
        include_forecast: 予測結果テーブルを含めるかどうか

    Returns:
        成功時True、失敗時False
    """
    from src.api.discord_bot import convert_df_for_discord
    from src.utils.db import (
        load_latest_prediction_timestamp,
        load_prediction_markets,
        load_prediction_results,
    )

    # 1. 完了メッセージを送信
    title = "✅ 日次パイプライン完了"

    message_lines = [
        f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if data_count is not None:
        message_lines.append(f"取得データ: {data_count}件")

    if prediction_markets:
        markets_str = "、".join(prediction_markets)
        message_lines.append(f"予測市場: {markets_str}")

    message = "\n".join(message_lines)

    success = send_webhook_notification(title, message, color=0x00FF00)

    # 2. 予測結果テーブルを送信
    if include_forecast:
        try:
            latest_ts = load_latest_prediction_timestamp()
            if latest_ts:
                markets = load_prediction_markets(latest_ts)

                if markets:
                    # Top10送信
                    for market in sorted(markets):
                        df = load_prediction_results(
                            predicted_at=latest_ts, market=market, top_n=10
                        )
                        if df is not None and not df.empty:
                            df = convert_df_for_discord(df)
                            table_text = df.to_string(index=False)
                            msg = f"=== {market} 差異割合上位10銘柄 ===\n```text\n{table_text}\n```"

                            # Discordメッセージ長制限対応（テキストメッセージは4000文字）
                            max_length = 3800
                            for i in range(0, len(msg), max_length):
                                send_webhook_text(msg[i : i + max_length])

                    # ワースト10送信
                    for market in sorted(markets):
                        df = load_prediction_results(
                            predicted_at=latest_ts, market=market, worst_n=10
                        )
                        if df is not None and not df.empty:
                            df = convert_df_for_discord(df)
                            table_text = df.to_string(index=False)
                            msg = f"=== {market} 差異割合ワースト10銘柄 ===\n```text\n{table_text}\n```"

                            # Discordメッセージ長制限対応
                            max_length = 3800
                            for i in range(0, len(msg), max_length):
                                send_webhook_text(msg[i : i + max_length])

        except Exception as e:
            logger.error(f"予測結果テーブル送信失敗: {e}")

    return success


def send_daily_pipeline_error(error_message: str) -> bool:
    """
    日次パイプラインエラー通知

    Args:
        error_message: エラーメッセージ

    Returns:
        成功時True、失敗時False
    """
    title = "❌ 日次パイプライン失敗"

    message = f"エラー: {error_message}\n時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return send_webhook_notification(title, message, color=0xFF0000)
