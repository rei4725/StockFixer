"""Claude Extended Thinking の推論ログ保存。

claude_agent.py から分離（ファイル行数ゲート対応）。
"""

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_reasoning_log(run_id: str, market: str, thinking_text: str, summary: str) -> None:
    """Extended thinking の推論ログを claude_reasoning テーブルに保存する。"""  # noqa: D401
    try:
        with _db_connection() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS claude_reasoning (
                    run_id      VARCHAR PRIMARY KEY,
                    market      VARCHAR,
                    thinking    TEXT,
                    summary     TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            con.execute(
                """
                INSERT INTO claude_reasoning (run_id, market, thinking, summary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    market = EXCLUDED.market,
                    thinking = EXCLUDED.thinking,
                    summary = EXCLUDED.summary
                """,
                [run_id, market, thinking_text, summary],
            )
    except Exception:
        logger.warning("[claude_agent] 推論ログ保存に失敗しました", exc_info=True)
