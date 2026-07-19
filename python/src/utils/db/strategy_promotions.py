"""
戦略ファクトリー自動昇格ループ: 昇格記録テーブル操作

テーブル:
    strategy_promotions … マージされた戦略ファクトリー由来 PR の昇格台帳。
    ロールバック監視ジョブが実績を追跡する際の対象一覧としても使う。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DDL_STRATEGY_PROMOTIONS = """
CREATE TABLE IF NOT EXISTS strategy_promotions (
    pr_number               INTEGER   NOT NULL,
    merge_commit_hash       VARCHAR   NOT NULL,
    rule_or_feature_id      VARCHAR   NOT NULL,
    promoted_at              TIMESTAMP NOT NULL,
    pre_promotion_baseline  DOUBLE    NOT NULL,
    status                  VARCHAR   NOT NULL DEFAULT 'active',
    PRIMARY KEY (pr_number)
)
"""


def ensure_strategy_promotions_table() -> None:
    with _db_connection() as con:
        con.execute(_DDL_STRATEGY_PROMOTIONS)


def save_strategy_promotion(
    pr_number: int,
    merge_commit_hash: str,
    rule_or_feature_id: str,
    pre_promotion_baseline: float,
    promoted_at: Optional[datetime] = None,
) -> None:
    """マージ検出した戦略昇格を記録する（同一 pr_number は置換）。"""
    ensure_strategy_promotions_table()
    if promoted_at is None:
        promoted_at = datetime.now()
    with _db_connection() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO strategy_promotions (
                pr_number, merge_commit_hash, rule_or_feature_id,
                promoted_at, pre_promotion_baseline, status
            )
            VALUES (?, ?, ?, ?, ?, 'active')
            """,
            [pr_number, merge_commit_hash, rule_or_feature_id, promoted_at, pre_promotion_baseline],
        )
    logger.info(
        "戦略昇格記録: pr=%s hash=%s baseline=%.3f",
        pr_number,
        rule_or_feature_id,
        pre_promotion_baseline,
    )


def promotion_exists(pr_number: int) -> bool:
    """指定 PR がすでに記録済みかどうかを返す（マージ検知ジョブの重複防止用）。"""
    ensure_strategy_promotions_table()
    with _db_connection() as con:
        row = con.execute(
            "SELECT 1 FROM strategy_promotions WHERE pr_number = ?", [pr_number]
        ).fetchone()
    return row is not None


def load_active_promotions() -> pd.DataFrame:
    """status='active' の昇格レコードを返す（ロールバック監視ジョブの対象一覧）。"""
    ensure_strategy_promotions_table()
    with _db_connection() as con:
        try:
            return con.execute(
                "SELECT * FROM strategy_promotions "
                "WHERE status = 'active' ORDER BY promoted_at DESC"
            ).fetchdf()
        except Exception as e:
            logger.error("strategy_promotions 読み込み失敗: %s", e, exc_info=True)
            return pd.DataFrame()


def mark_promotion_rolled_back(pr_number: int) -> None:
    """指定 PR の昇格をロールバック済みとしてマークする。"""
    ensure_strategy_promotions_table()
    with _db_connection() as con:
        con.execute(
            "UPDATE strategy_promotions SET status = 'rolled_back' WHERE pr_number = ?",
            [pr_number],
        )
    logger.info("戦略昇格をロールバック済みにマーク: pr=%s", pr_number)
