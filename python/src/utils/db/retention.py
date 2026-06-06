"""診断ログテーブルの retention（保持期間管理）。

shap_values / feature_selection_log は学習のたびに大量行（数十万行/日）が増え、
DB 肥大化の主因になる。読み出し側はいずれも各 (market, symbol[, model]) の
``MAX(trained_at)`` しか参照しないため、古い行の削除は安全。

ただし「保持日数より古い＝全削除」では、長期間学習されていない銘柄の最新行まで
消えて読み出しが壊れる。そこで **各グループの最新 trained_at は常に残し**、
それ以外で保持日数より古い行のみを削除する。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# (テーブル名, グループキー列) — グループの最新 trained_at は常に保持する
_LOG_TABLES: list[tuple[str, tuple[str, ...]]] = [
    ("shap_values", ("market", "symbol", "model_name")),
    ("feature_selection_log", ("market", "symbol", "model_name")),
]


def _cutoff(retention_days: int, now: datetime | None = None) -> str:
    """trained_at 形式（YYYYMMDD_HHMMSS）の cutoff 文字列を返す。"""
    base = now or datetime.now(timezone.utc)
    return (base - timedelta(days=retention_days)).strftime("%Y%m%d_%H%M%S")


def purge_old_training_logs(
    con: Any,
    retention_days: int,
    now: datetime | None = None,
) -> dict[str, int]:
    """古い shap_values / feature_selection_log 行を削除し、削除件数を返す。

    Args:
        con: DuckDB 接続（書き込み可能なもの）。
        retention_days: 保持日数。これより古い行を削除（各グループの最新を除く）。
        now: 基準時刻（テスト用。省略時は現在 UTC）。

    Returns:
        {テーブル名: 削除行数}
    """
    if retention_days < 0:
        raise ValueError("retention_days は 0 以上が必要です")

    cutoff = _cutoff(retention_days, now)
    deleted: dict[str, int] = {}

    for table, keys in _LOG_TABLES:
        key_join = " AND ".join(f't2.{k} = "{table}".{k}' for k in keys)
        # 保持日数より古く、かつ自グループの最新 trained_at ではない行のみ削除
        where = (
            f"trained_at < ? "
            f'AND trained_at < (SELECT MAX(t2.trained_at) FROM "{table}" t2 WHERE {key_join})'
        )
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {where}', [cutoff]).fetchone()[0]
            if n:
                con.execute(f'DELETE FROM "{table}" WHERE {where}', [cutoff])
            deleted[table] = int(n)
            logger.info("retention: %s から %d 行削除 (cutoff=%s)", table, n, cutoff)
        except Exception as e:
            logger.warning("retention 失敗 %s: %s", table, e, exc_info=True)
            deleted[table] = 0

    return deleted
