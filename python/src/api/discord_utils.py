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
        logger.warning("DISCORD_WEBHOOK_URLが環境変数に設定されていません。" "Webhook通知をスキップします。")
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
        logger.warning("DISCORD_WEBHOOK_URLが環境変数に設定されていません。" "Webhook通知をスキップします。")
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


def send_webhook_file(file_path: str, title: str = "") -> bool:
    """
    ファイル（PNG等）を Discord Webhook にアップロードして送信する。

    Args:
        file_path: 送信するファイルの絶対パス
        title: 添付メッセージ（省略可）

    Returns:
        成功時True、失敗時False
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("DISCORD_WEBHOOK_URLが環境変数に設定されていません。ファイル送信をスキップします。")
        return False

    if not os.path.exists(file_path):
        logger.warning(f"送信対象ファイルが存在しません: {file_path}")
        return False

    try:
        with open(file_path, "rb") as f:
            filename = os.path.basename(file_path)
            payload = {"content": title} if title else {}
            response = requests.post(
                webhook_url,
                data=payload,
                files={"file": (filename, f)},
                timeout=30,
            )
        response.raise_for_status()
        logger.info(f"Discordファイル送信成功: {filename}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Discordファイル送信失敗: {e}")
        return False


def send_drift_alert(summary_df, horizon: int = 1, threshold: float = 0.45) -> bool:
    """
    モデルドリフト警告を Discord Webhook に送信する。

    方向正解率が threshold 以下の銘柄が存在する場合にのみ送信する。

    Args:
        summary_df: load_drift_summary() の戻り値 (DataFrame)
        horizon: 対象ホライズン（メッセージ表示用）
        threshold: 警告する方向正解率の閾値（デフォルト 0.45 = 45%）

    Returns:
        送信成功時 True、送信不要または失敗時 False
    """
    import pandas as pd

    if summary_df is None or (isinstance(summary_df, pd.DataFrame) and summary_df.empty):
        return False

    drift_rows = summary_df[summary_df["direction_accuracy"] <= threshold]
    if drift_rows.empty:
        logger.info(f"ドリフト警告なし (horizon={horizon}d, 閾値={threshold:.0%})")
        return False

    lines = [f"**[モデルドリフト警告] horizon={horizon}d (方向正解率 ≤ {threshold:.0%})**\n"]
    for _, row in drift_rows.iterrows():
        acc = row.get("direction_accuracy", 0)
        err = row.get("mean_abs_error", 0)
        n = int(row.get("n_samples", 0))
        lines.append(
            f"• `{row['market']}/{row['symbol']}` " f"正解率={acc:.1%}, 平均誤差={err:.4f}, N={n}"
        )

    message = "\n".join(lines)
    return send_webhook_text(message)


def send_weekly_report(accuracy_df=None, horizon: int = 1) -> bool:
    """
    週次パフォーマンスレポートを Discord Webhook に送信する。

    直近の方向正解率・平均誤差を銘柄ごとに集計してレポートする。

    Args:
        accuracy_df: load_drift_summary() の戻り値 DataFrame（None の場合はDB から取得）
        horizon: 対象ホライズン

    Returns:
        成功時 True、失敗時 False
    """
    import pandas as pd

    from src.utils.db import load_drift_summary

    if accuracy_df is None or (isinstance(accuracy_df, pd.DataFrame) and accuracy_df.empty):
        accuracy_df = load_drift_summary(horizon=horizon)

    if accuracy_df is None or (isinstance(accuracy_df, pd.DataFrame) and accuracy_df.empty):
        logger.info("週次レポート: 精度データなし")
        return False

    from datetime import datetime

    now = datetime.now().strftime("%Y/%m/%d")
    lines = [f"**[週次パフォーマンスレポート] {now} (horizon={horizon}d)**\n"]

    # 方向正解率でソート（低い順 = 要注意銘柄を先頭に）
    df_sorted = accuracy_df.sort_values("direction_accuracy", ascending=True)
    lines.append("**銘柄別 方向正解率（低い順）**")
    for _, row in df_sorted.head(20).iterrows():
        acc = row.get("direction_accuracy", 0)
        err = row.get("mean_abs_error", 0)
        n = int(row.get("n_samples", 0))
        flag = " ⚠️" if acc <= 0.45 else ""
        lines.append(
            f"• `{row['market']}/{row['symbol']}` " f"正解率={acc:.1%}, 平均誤差={err:.4f}, N={n}{flag}"
        )

    # 全体サマリー
    mean_acc = accuracy_df["direction_accuracy"].mean()
    lines.append(f"\n**全体平均正解率**: {mean_acc:.1%} ({len(accuracy_df)}銘柄)")

    message = "\n".join(lines)
    # Discord の 2000 文字制限に対応した分割送信
    chunks: list[str] = []
    current = ""
    for line in message.split("\n"):
        if len(current) + len(line) + 1 > 1900:
            chunks.append(current)
            current = line
        else:
            current = (current + "\n" + line) if current else line
    if current:
        chunks.append(current)

    success = True
    for chunk in chunks:
        if not send_webhook_text(chunk):
            success = False
    return success
