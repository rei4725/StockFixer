"""
銘柄マスタ（コード → 社名）テーブル操作

テーブル:
    symbol_master … (market, symbol) → 日本語社名・英語名・業種。
    jp は JPX 一覧の日本語社名、us は Nasdaq Symbol Directory の英語名が入る。
    DDL はマイグレーション 0007 が作る。外部の読み手は SELECT だけで社名を引ける。
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Protocol, Sequence, Tuple

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


# ---- 通知などの表示用: コードに社名を添える -----------------------------------------
# 社名は週に1回しか変わらないので、プロセスの中でマーケットごとに一度読んで持っておく。
# DB が読めないときは空の表を持ち、表示はコードだけに戻す（通知は止めない）。
_DISPLAY_NAME_TTL_SEC = 6 * 60 * 60
_DISPLAY_NAME_MAX_LEN = 24
_JP_CODE = re.compile(r"^\d{3}[0-9A-Z]$")
_display_name_cache: Dict[str, Tuple[float, Dict[str, str]]] = {}


def load_symbol_display_names(market: str) -> Dict[str, str]:
    """指定マーケットの {symbol: 表示名} を返す。日本語社名を優先し、無ければ英語名。"""
    with _db_connection() as con:
        rows = con.execute(
            "SELECT symbol, COALESCE(name_ja, name_en) FROM symbol_master "
            "WHERE market = %s AND COALESCE(name_ja, name_en) IS NOT NULL",
            [market],
        ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def _cached_display_names(market: str) -> Dict[str, str]:
    now = time.monotonic()
    hit = _display_name_cache.get(market)
    if hit is not None and now - hit[0] < _DISPLAY_NAME_TTL_SEC:
        return hit[1]
    try:
        names = load_symbol_display_names(market)
    except Exception as e:
        logger.warning("symbol_master 読み込み失敗（表示はコードだけにする）: %s", e, exc_info=True)
        names = {}
    _display_name_cache[market] = (now, names)
    return names


def clear_symbol_display_name_cache() -> None:
    """表示名のキャッシュを捨てる（銘柄マスタ更新の直後とテスト用）。"""
    _display_name_cache.clear()


def guess_market(symbol: str) -> str:
    """market が手元に無い呼び出し元のために、コードの形からマーケットを推す。

    東証のコードは 4 桁（2024 年からは "130A" のように末尾に英字も入る）、それ以外は us とみなす。
    """
    code = symbol.upper()
    if code.endswith(".T"):
        return "jp"
    return "jp" if _JP_CODE.match(code) else "us"


def symbol_display_name(market: Optional[str], symbol: str) -> Optional[str]:
    """銘柄の表示名（長いものは詰める）を返す。未登録や DB が読めないときは None。"""
    code = str(symbol)
    if code.upper().endswith(".T"):
        code = code[:-2]
    name = _cached_display_names((market or guess_market(code)).lower()).get(code)
    if not name:
        return None
    if len(name) > _DISPLAY_NAME_MAX_LEN:
        name = name[: _DISPLAY_NAME_MAX_LEN - 1] + "…"
    return name


def symbol_label(market: Optional[str], symbol: str) -> str:
    """コードに社名を添えた表示を返す（例: 7203 トヨタ自動車 / AAPL Apple Inc.）。名前が無ければコードだけ。"""
    name = symbol_display_name(market, symbol)
    return f"{symbol} {name}" if name else str(symbol)


def symbol_name_suffix(market: Optional[str], symbol: str) -> str:
    """既存の表示（`jp/7203` など）の後ろに付ける " 社名"。名前が無ければ空文字。"""
    name = symbol_display_name(market, symbol)
    return f" {name}" if name else ""
