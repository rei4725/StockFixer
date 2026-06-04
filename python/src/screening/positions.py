"""月次ライブ・スクリーンの保有ポジション台帳（JSON 永続化）

半自動運用（売買は人間が最終判断）のため、人間が建てたと宣言したポジションを
``config/multibagger_positions.json`` に永続化し、月次ジョブ(#434)が状態
（保有/段階利確/撤退）を更新する。

DuckDB ではなく JSON を採用する理由:
- 件数が少なく（保有数 = 数十件規模）、人手で確認・編集しやすい。
- CLAUDE.md の「DuckDB 並行書き込み禁止」制約を避けられる（別プロセスのジョブと
  競合しない）。

依存は ``utils`` のみ（他 BC を import しない純粋 BC）。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from src.screening.types import Position
from src.utils.data_path_utils import ensure_dir, get_multibagger_positions_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _today() -> str:
    """評価日（UTC 基準の YYYY-MM-DD）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_raw() -> list[dict]:
    """JSON から生レコード（dict）のリストを読み込む。ファイルが無ければ空。"""
    path = get_multibagger_positions_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("ポジション台帳の読み込み失敗: %s", e, exc_info=True)
        return []
    if not isinstance(data, list):
        logger.error("ポジション台帳の形式が不正です（list を期待）")
        return []
    return data


def _save_raw(records: list[dict]) -> None:
    """生レコードのリストを JSON に保存する（逐次書き込み）。"""
    path = get_multibagger_positions_path()
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _to_position(rec: dict) -> Position:
    """生レコードを Position に変換する（未知キーは無視）。"""
    fields = Position.__dataclass_fields__
    return Position(**{k: v for k, v in rec.items() if k in fields})


def list_positions(status: Optional[str] = None) -> list[Position]:
    """全ポジションを返す。status を指定するとその状態のみに絞る。"""
    positions = [_to_position(r) for r in _load_raw()]
    if status is not None:
        positions = [p for p in positions if p.status == status]
    return positions


def get_position(market: str, symbol: str) -> Optional[Position]:
    """(market, symbol) の open ポジションを返す（無ければ None）。"""
    for p in list_positions(status="open"):
        if p.market == market and p.symbol == symbol:
            return p
    return None


def add_position(market: str, symbol: str, entry_date: str, entry_price: float) -> Position:
    """新規ポジションを台帳に追加する。

    同一 (market, symbol) の open ポジションが既にある場合は追加せず既存を返す。
    """
    existing = get_position(market, symbol)
    if existing is not None:
        logger.warning("既に open ポジションが存在します: %s/%s", market, symbol)
        return existing

    pos = Position(
        market=market,
        symbol=symbol,
        entry_date=entry_date,
        entry_price=float(entry_price),
        last_evaluated=_today(),
    )
    records = _load_raw()
    records.append(asdict(pos))
    _save_raw(records)
    logger.info("ポジション追加: %s/%s @ %s (%s)", market, symbol, entry_price, entry_date)
    return pos


def update_position(market: str, symbol: str, **fields) -> Optional[Position]:
    """open ポジションのフィールドを更新する。対象が無ければ None。"""
    records = _load_raw()
    updated: Optional[Position] = None
    for rec in records:
        is_target = (
            rec.get("market") == market
            and rec.get("symbol") == symbol
            and rec.get("status") == "open"
        )
        if is_target:
            for key, value in fields.items():
                if key in Position.__dataclass_fields__:
                    rec[key] = value
            rec["last_evaluated"] = _today()
            updated = _to_position(rec)
            break
    if updated is None:
        logger.warning("更新対象の open ポジションが見つかりません: %s/%s", market, symbol)
        return None
    _save_raw(records)
    return updated


def close_position(
    market: str,
    symbol: str,
    close_price: float,
    reason: str,
    close_date: Optional[str] = None,
) -> Optional[Position]:
    """open ポジションを撤退（closed）として確定する。対象が無ければ None。"""
    pos = get_position(market, symbol)
    if pos is None:
        logger.warning("撤退対象の open ポジションが見つかりません: %s/%s", market, symbol)
        return None
    multiple = float(close_price) / pos.entry_price if pos.entry_price else 0.0
    return update_position(
        market,
        symbol,
        status="closed",
        held_fraction=0.0,
        last_action="exit",
        last_reason=reason,
        last_multiple=multiple,
        last_evaluated=close_date or _today(),
    )
