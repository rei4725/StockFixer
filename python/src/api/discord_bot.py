import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN") 
INTENTS = discord.Intents.default()
INTENTS.message_content = True

bot = commands.Bot(command_prefix="!", intents=INTENTS)

@bot.event
async def on_ready():
    print(f"Bot起動完了: {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content == "/Next10":
        import pandas as pd
        import os

        # CSV読み込みのみ
        # python/results/top10_diff_stocks.csv を絶対パスで参照
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/top10_diff_stocks.csv"))
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            msg = "=== 差異割合上位10銘柄 ===\n"
            msg += df.to_string(index=False)
            for chunk in [msg[i:i+1900] for i in range(0, len(msg), 1900)]:
                await message.channel.send(f"```{chunk}```")
        else:
            await message.channel.send(f"結果CSVが見つかりませんでした。\nパス: {csv_path}")
    else:
        # 受信したメッセージ内容をそのまま返信
        await message.channel.send(f"受信: {message.content}")

    # コマンド処理も有効化
    await bot.process_commands(message)

if __name__ == "__main__":
    if not TOKEN:
        print("DISCORD_BOT_TOKEN環境変数が設定されていません。")
    else:
        bot.run(TOKEN)
