import json
import os

import discord
import pandas as pd
from discord.ext import commands
from discord.utils import escape_markdown
from dotenv import load_dotenv

from src.utils.data_path_utils import get_monitor_list_path, get_results_dir
from src.utils.db import (
    load_latest_prediction_timestamp,
    load_prediction_markets,
    load_prediction_results,
)

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
INTENTS = discord.Intents.default()
INTENTS.message_content = True

bot = commands.Bot(command_prefix="!", intents=INTENTS)


def get_top10_diff_stocks_df(market: str, rank_type: str, predicted_at: str = None) -> pd.DataFrame:
    """DBから予測結果を取得する（top_n/worst_nでフィルタ）"""
    if rank_type == "top10":
        df = load_prediction_results(predicted_at=predicted_at, market=market, top_n=10)
    elif rank_type == "worst10":
        df = load_prediction_results(predicted_at=predicted_at, market=market, worst_n=10)
    else:
        df = load_prediction_results(predicted_at=predicted_at, market=market)
    if df is None:
        return pd.DataFrame()
    return df


def convert_df_for_discord(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    # 列名変換・順序統一
    columns_map = {
        "symbol": "シンボル",
        "current_price": "現在値",
        "avg_pred_price": "予想終値",
        "diff_ratio": "予想変化率",
        "予想値": "予想終値",
    }
    col_order = ["シンボル", "現在値", "予想終値", "予想変化率"]
    df = df.rename(columns=columns_map)
    # 予想変化率がなければ計算
    if "現在値" in df.columns and "予想終値" in df.columns and "予想変化率" not in df.columns:
        try:
            df["予想変化率"] = (df["予想終値"].astype(float) - df["現在値"].astype(float)) / df["現在値"].astype(
                float
            )
        except Exception:
            df["予想変化率"] = ""
    # 数値列変換
    if "現在値" in df.columns:
        df["現在値"] = df["現在値"].apply(
            lambda x: np.floor(float(x) * 1000) / 1000 if pd.notnull(x) else x
        )
    if "予想終値" in df.columns:
        df["予想終値"] = df["予想終値"].apply(
            lambda x: np.floor(float(x) * 1000) / 1000 if pd.notnull(x) else x
        )
    if "予想変化率" in df.columns:

        def format_percent(val):
            try:
                v = float(val)
                sign = "+" if v >= 0 else ""
                return f"{sign}{v*100:.2g}%"
            except Exception:
                return val

        df["予想変化率"] = df["予想変化率"].apply(format_percent)
    # 列順並べ替え
    df = df[[c for c in col_order if c in df.columns]]
    return df


def get_top10_diff_stocks_message(market: str, rank_type: str, predicted_at: str = None) -> str:
    """DBから予測結果を取得してDiscord表示用テキストに変換する"""
    df = get_top10_diff_stocks_df(market, rank_type, predicted_at)
    if df.empty:
        return ""
    df = convert_df_for_discord(df)
    table_text = df.to_string(index=False)
    return table_text


async def handle_forecast_command(message):
    # 最新の予測結果をDBから取得
    latest_ts = load_latest_prediction_timestamp()
    if latest_ts is None:
        await message.channel.send(escape_markdown("予測結果が見つかりませんでした。"), allowed_mentions=None)
        return

    markets = load_prediction_markets(latest_ts)
    if not markets:
        await message.channel.send(escape_markdown("予測結果が見つかりませんでした。"), allowed_mentions=None)
        return

    # Top10送信
    for market in sorted(markets):
        table_text = get_top10_diff_stocks_message(market, "top10", latest_ts)
        if not table_text:
            continue
        msg = f"=== {market} 差異割合上位10銘柄 ===\n```text\n{table_text}\n```"
        # Discordメッセージ長制限対応
        max_length = 1900
        for i in range(0, len(msg), max_length):
            await message.channel.send(msg[i : i + max_length])

    # ワースト10送信
    for market in sorted(markets):
        table_text = get_top10_diff_stocks_message(market, "worst10", latest_ts)
        if not table_text:
            continue
        msg = f"=== {market} 差異割合ワースト10銘柄 ===\n```text\n{table_text}\n```"
        max_length = 1900
        for i in range(0, len(msg), max_length):
            await message.channel.send(msg[i : i + max_length])


@bot.event
async def on_ready():
    print(f"Bot起動完了: {bot.user}")


def get_watchlist_prediction_text():
    import csv

    from src.models.predict_single_stock import predict_single_stock

    watchlist_path = get_monitor_list_path()
    rows = []
    try:
        with open(watchlist_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                market, symbol = row[0], row[1]
                result_df = predict_single_stock(market, symbol)
                if result_df is None or result_df.empty:
                    rows.append([symbol, "-", "-", "-"])
                else:
                    r = result_df.iloc[0]
                    try:
                        diff_ratio = (
                            float(r["avg_pred_price"]) - float(r["current_price"])
                        ) / float(r["current_price"])
                    except Exception:
                        diff_ratio = "-"
                    rows.append(
                        [
                            str(r["symbol"]),
                            f'{r["current_price"]:.2f}',
                            f'{r["avg_pred_price"]:.2f}',
                            diff_ratio,
                        ]
                    )
    except Exception as e:
        return f"[エラー] 監視対象予測処理で例外: {e}"

    # DataFrame化して共通変換部品で整形
    df = pd.DataFrame(rows, columns=["symbol", "current_price", "avg_pred_price", "diff_ratio"])
    df = convert_df_for_discord(df)
    table_text = df.to_string(index=False)
    return table_text


async def handle_watchnext_command(message):
    text = get_watchlist_prediction_text()
    # Discordメッセージ長制限対応
    max_length = 1900
    if len(text) > max_length:
        text = text[:max_length]

    msg = f"=== 監視対象銘柄 ===\n```text\n{text}\n```"
    await message.channel.send(msg)


async def handle_signal_command(message, parts: list):
    """
    /signal <symbol> [market] [--explain]
    指定銘柄の最新予測シグナルを表示する。
    --explain を付けると SHAP による特徴量寄与度トップ5を追加表示する。
    """
    if len(parts) < 2:
        await message.channel.send(
            escape_markdown(
                "使い方: /signal <銘柄コード> [market] [--explain]  例: /signal AAPL us --explain"
            ),
            allowed_mentions=None,
        )
        return

    symbol = parts[1].upper()
    market = parts[2].lower() if len(parts) >= 3 and not parts[2].startswith("--") else "us"
    explain = "--explain" in parts

    from src.models.predict_single_stock import explain_prediction_shap, predict_single_stock

    await message.channel.send(escape_markdown(f"計算中... {market}/{symbol}"), allowed_mentions=None)
    result_df = predict_single_stock(market, symbol)
    if result_df is None or result_df.empty:
        await message.channel.send(
            escape_markdown(f"モデルが見つかりませんでした: {market}/{symbol}"),
            allowed_mentions=None,
        )
        return

    r = result_df.iloc[0]
    diff_ratio = float(r["diff_ratio"])
    current_price = float(r["current_price"])
    pred_price = float(r["avg_pred_price"])
    model_count = int(r["model_count"])

    if diff_ratio > 0.005:
        signal = "⬆️ Buy"
    elif diff_ratio < -0.005:
        signal = "⬇️ Sell"
    else:
        signal = "⏺️ Hold"

    import math

    current_disp = math.floor(current_price * 1000) / 1000
    pred_disp = math.floor(pred_price * 1000) / 1000
    change_pct = f"{diff_ratio * 100:.2g}%"

    lines = [
        f"=== {market.upper()} / {symbol} ===",
        f"現在値  : {current_disp}",
        f"予想終値: {pred_disp}",
        f"予想変化率: {change_pct}",
        f"シグナル : {signal}",
        f"使用モデル: {model_count}モデルの平均",
    ]

    if explain:
        shap_result = explain_prediction_shap(market, symbol, top_n=5)
        if shap_result:
            lines.append("")
            lines.append(f"[SHAP 寄与度 Top5 - 方向:{shap_result['direction']}]")
            for feat in shap_result["top_features"]:
                arrow = "▲" if feat["shap_value"] > 0 else "▼"
                lines.append(f"  {arrow} {feat['feature'][:30]:<30}  {feat['shap_value']:+.5f}")
        else:
            lines.append("(SHAP 計算スキップ: モデルまたはライブラリが未対応)")

    msg = "```text\n" + "\n".join(lines) + "\n```"
    await message.channel.send(msg)


async def handle_status_command(message):
    """
    /status
    スケジューラの最終実行時刻を表示する。
    """
    state_file = os.path.join(get_results_dir(), "scheduler_queue_state.json")
    if not os.path.exists(state_file):
        await message.channel.send(
            escape_markdown("スケジューラ状態ファイルが見つかりません。スケジューラが起動されていない可能性があります。"),
            allowed_mentions=None,
        )
        return

    try:
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        await message.channel.send(
            escape_markdown(f"状態ファイルの読み込みに失敗しました: {e}"),
            allowed_mentions=None,
        )
        return

    events = state.get("events", [])

    # job_id ごとの最終実行を集計
    last_runs: dict = {}
    last_status: dict = {}
    for event in reversed(events):
        job_id = event.get("job_id", "")
        if job_id and job_id not in last_runs:
            last_runs[job_id] = event.get("finished_at") or event.get("started_at") or "不明"
            last_status[job_id] = event.get("status", "不明")

    job_labels = {
        "daily_pipeline": "日次 (daily_pipeline)",
        "weekly_model_training": "週次 (weekly_model_training)",
    }

    lines = ["=== スケジューラ状態 ==="]
    for job_id, label in job_labels.items():
        ts = last_runs.get(job_id, "実行履歴なし")
        st = last_status.get(job_id, "-")
        if len(ts) >= 19:
            ts = ts[:19].replace("T", " ")
        lines.append(f"{label}")
        lines.append(f"  最終実行: {ts}  [状態: {st}]")

    msg = "```text\n" + "\n".join(lines) + "\n```"
    await message.channel.send(msg)


@bot.event
async def on_message(message):
    try:
        if message.author.bot:
            return

        parts = message.content.split()
        command = parts[0] if parts else ""

        if command == "/forecast":
            await handle_forecast_command(message)
        elif command == "/WatchNext":
            await handle_watchnext_command(message)
        elif command == "/signal":
            await handle_signal_command(message, parts)
        elif command == "/status":
            await handle_status_command(message)
        else:
            await message.channel.send(
                escape_markdown(f"受信: {message.content}"), allowed_mentions=None
            )

        await bot.process_commands(message)
    except Exception as e:
        await message.channel.send(escape_markdown("エラーが発生しました"), allowed_mentions=None)
        print(f"Error in on_message: {e}")


if __name__ == "__main__":
    if not TOKEN:
        print("DISCORD_BOT_TOKEN環境変数が設定されていません。")
    else:
        bot.run(TOKEN)
