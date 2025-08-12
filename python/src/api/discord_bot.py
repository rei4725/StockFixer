import os
import sys
import pandas as pd
import discord
from discord.ext import commands
from dotenv import load_dotenv
from src.utils.df_to_string import df_to_pretty_string

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN") 
INTENTS = discord.Intents.default()
INTENTS.message_content = True

bot = commands.Bot(command_prefix="!", intents=INTENTS)

def get_top10_diff_stocks_message(csv_path: str) -> str:
    if not os.path.exists(csv_path):
        return f"結果CSVが見つかりませんでした。\nパス: {csv_path}"
    df = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path)
    header = f"=== 差異割合上位10銘柄 ({filename}) ==="
    return df_to_pretty_string(df, header=header)

@bot.event
async def on_ready():
    print(f"Bot起動完了: {bot.user}")

@bot.event
async def on_message(message):
    try:
        if message.author.bot:
            return

        if message.content == "/Next10":
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/top10_diff_stocks.csv"))
            msg = get_top10_diff_stocks_message(csv_path)
            await message.channel.send(f"```{msg}```")
        else:
            # 受信したメッセージ内容をそのまま返信
            await message.channel.send(f"受信: {message.content}")

        # コマンド処理も有効化
        await bot.process_commands(message)
    except Exception as e:
        await message.channel.send("エラーが発生しました")
        print(f"Error in on_message: {e}")

if __name__ == "__main__":
    if not TOKEN:
        print("DISCORD_BOT_TOKEN環境変数が設定されていません。")
    else:
        bot.run(TOKEN)
