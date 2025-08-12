from src.api.discord_bot import bot, TOKEN

if __name__ == "__main__":
    if not TOKEN:
        print("DISCORD_BOT_TOKEN環境変数が設定されていません。")
    else:
        bot.run(TOKEN)
