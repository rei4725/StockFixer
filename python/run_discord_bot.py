import sys

from src.api.discord_bot import TOKEN, bot
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    if not TOKEN:
        logger.critical("DISCORD_BOT_TOKEN環境変数が設定されていません。")
        sys.exit(1)
    try:
        bot.run(TOKEN)
    except Exception as e:
        logger.critical(f"Discord Bot 異常終了: {e}", exc_info=True)
        sys.exit(1)
