"""
銘柄マスタ（コード → 社名）テーブル操作

テーブル:
    symbol_master … (market, symbol) → 日本語社名・英語名・業種。
    DDL はマイグレーション 0007 が作る。外部の読み手は SELECT だけで社名を引ける。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Protocol, Sequence

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _SymbolNameRow(Protocol):
    """保存する1行の形（market_data.types.SymbolName が満たす）。"""

    market: str
    symbol: str
    name_ja: Optional[str]
    name_en: Optional[str]
    sector: Optional[str]
    source: str


def upsert_symbol_names(
    records: Sequence[_SymbolNameRow], updated_at: Optional[datetime] = None
) -> int:
    """対応表を (market, symbol) 単位で置換保存する。Returns: 保存件数"""
    if not records:
        return 0
    if updated_at is None:
        updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    with _db_connection() as con:
        for r in records:
            con.execute(
                """
                INSERT INTO symbol_master
                    (market, symbol, name_ja, name_en, sector, source, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (market, symbol) DO UPDATE SET
                    name_ja = EXCLUDED.name_ja,
                    name_en = EXCLUDED.name_en,
                    sector = EXCLUDED.sector,
                    source = EXCLUDED.source,
                    updated_at = EXCLUDED.updated_at
                """,
                [r.market, r.symbol, r.name_ja, r.name_en, r.sector, r.source, updated_at],
            )
    logger.info("symbol_master 保存: %d 件", len(records))
    return len(records)


def load_symbol_names(market: str) -> Dict[str, str]:
    """指定マーケットの {symbol: name_ja} を返す（name_ja が無い行は除く）。"""
    with _db_connection() as con:
        rows = con.execute(
            "SELECT symbol, name_ja FROM symbol_master "
            "WHERE market = %s AND name_ja IS NOT NULL",
            [market],
        ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def get_symbol_name(market: str, symbol: str) -> Optional[str]:
    """銘柄コードから日本語社名を返す。未登録なら None。"""
    with _db_connection() as con:
        row = con.execute(
            "SELECT name_ja FROM symbol_master WHERE market = %s AND symbol = %s",
            [market, symbol],
        ).fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])
