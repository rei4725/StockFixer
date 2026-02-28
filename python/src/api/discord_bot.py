import os
import sys
import pandas as pd
import discord
from discord.ext import commands
from dotenv import load_dotenv
from discord.utils import escape_markdown
from src.utils.data_path_utils import get_monitor_list_path
from src.utils.db import load_prediction_results, load_latest_prediction_timestamp, load_prediction_markets

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN") 
INTENTS = discord.Intents.default()
INTENTS.message_content = True

bot = commands.Bot(command_prefix="!", intents=INTENTS)

def get_top10_diff_stocks_df(market: str, rank_type: str, run_timestamp: str = None) -> pd.DataFrame:
    """DBから予測結果を取得する"""
    df = load_prediction_results(run_timestamp=run_timestamp, market=market, rank_type=rank_type)
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
        "予想値": "予想終値"
    }
    col_order = ["シンボル", "現在値", "予想終値", "予想変化率"]
    df = df.rename(columns=columns_map)
    # 予想変化率がなければ計算
    if "現在値" in df.columns and "予想終値" in df.columns and "予想変化率" not in df.columns:
        try:
            df["予想変化率"] = (df["予想終値"].astype(float) - df["現在値"].astype(float)) / df["現在値"].astype(float)
        except Exception:
            df["予想変化率"] = ""
    # 数値列変換
    if "現在値" in df.columns:
        df["現在値"] = df["現在値"].apply(lambda x: np.floor(float(x)*1000)/1000 if pd.notnull(x) else x)
    if "予想終値" in df.columns:
        df["予想終値"] = df["予想終値"].apply(lambda x: np.floor(float(x)*1000)/1000 if pd.notnull(x) else x)
    if "予想変化率" in df.columns:
        def format_percent(val):
            try:
                v = float(val)
                return f"{v*100:.2g}%"
            except:
                return val
        df["予想変化率"] = df["予想変化率"].apply(format_percent)
    # 列順並べ替え
    df = df[[c for c in col_order if c in df.columns]]
    return df

def get_top10_diff_stocks_message(market: str, rank_type: str, run_timestamp: str = None) -> str:
    """DBから予測結果を取得してDiscord表示用テキストに変換する"""
    df = get_top10_diff_stocks_df(market, rank_type, run_timestamp)
    if df.empty:
        return ""
    df = convert_df_for_discord(df)
    table_text = df.to_string(index=False)
    return table_text

async def handle_forecast_command(message):
    # 最新の予測結果をDBから取得
    latest_ts = load_latest_prediction_timestamp()
    if latest_ts is None:
        await message.channel.send(
            escape_markdown("予測結果が見つかりませんでした。"),
            allowed_mentions=None
        )
        return

    markets = load_prediction_markets(latest_ts)
    if not markets:
        await message.channel.send(
            escape_markdown("予測結果が見つかりませんでした。"),
            allowed_mentions=None
        )
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
            await message.channel.send(msg[i:i+max_length])

    # ワースト10送信
    for market in sorted(markets):
        table_text = get_top10_diff_stocks_message(market, "worst10", latest_ts)
        if not table_text:
            continue
        msg = f"=== {market} 差異割合ワースト10銘柄 ===\n```text\n{table_text}\n```"
        max_length = 1900
        for i in range(0, len(msg), max_length):
            await message.channel.send(msg[i:i+max_length])

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
                        diff_ratio = (float(r["avg_pred_price"]) - float(r["current_price"])) / float(r["current_price"])
                    except Exception:
                        diff_ratio = "-"
                    rows.append([
                        str(r["symbol"]),
                        f'{r["current_price"]:.2f}',
                        f'{r["avg_pred_price"]:.2f}',
                        diff_ratio
                    ])
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

@bot.event
async def on_message(message):
    try:
        if message.author.bot:
            return

        if message.content == "/forecast":
            await handle_forecast_command(message)
        elif message.content == "/WatchNext":
            await handle_watchnext_command(message)
        else:
            await message.channel.send(
                escape_markdown(f"受信: {message.content}"),
                allowed_mentions=None
            )

        await bot.process_commands(message)
    except Exception as e:
        await message.channel.send(
            escape_markdown("エラーが発生しました"),
            allowed_mentions=None
        )
        print(f"Error in on_message: {e}")

if __name__ == "__main__":
    if not TOKEN:
        print("DISCORD_BOT_TOKEN環境変数が設定されていません。")
    else:
        bot.run(TOKEN)
