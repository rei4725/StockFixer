"""
戦略ファクトリー用 DuckDB テーブル操作（#369 Phase 1）

テーブル:
    factory_runs … 夜間に評価した戦略仮説の台帳（重複試行防止 + DSR の累計試行数の源泉）

対照（is_control）仮説は毎晩再評価するため記録しない。候補仮説のみを保存する。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DDL_FACTORY_RUNS = """
CREATE TABLE IF NOT EXISTS factory_runs (
    hypothesis_hash  VARCHAR   NOT NULL,
    market           VARCHAR   NOT NULL,
    spec_json        VARCHAR   NOT NULL,
    sharpe_ratio     DOUBLE,
    win_rate         DOUBLE,
    num_trades       INTEGER,
    max_drawdown     DOUBLE,
    total_return     DOUBLE,
    dsr              DOUBLE,
    pbo              DOUBLE,
    gate_passed      BOOLEAN   NOT NULL,
    gate_reasons     VARCHAR,
    report_path      VARCHAR,
    evaluated_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hypothesis_hash)
)
"""


def ensure_factory_tables() -> None:
    with _db_connection() as con:
        con.execute(_DDL_FACTORY_RUNS)


def save_factory_run(
    hypothesis_hash: str,
    market: str,
    spec_json: str,
    sharpe_ratio: Optional[float],
    win_rate: Optional[float],
    num_trades: Optional[int],
    max_drawdown: Optional[float],
    total_return: Optional[float],
    dsr: Optional[float],
    pbo: Optional[float],
    gate_passed: bool,
    gate_reasons: Optional[str] = None,
    report_path: Optional[str] = None,
) -> None:
    """評価済み仮説を factory_runs に保存する（同一ハッシュは置換）。"""
    ensure_factory_tables()
    with _db_connection() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO factory_runs (
                hypothesis_hash, market, spec_json,
                sharpe_ratio, win_rate, num_trades, max_drawdown, total_return,
                dsr, pbo, gate_passed, gate_reasons, report_path, evaluated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                hypothesis_hash,
                market,
                spec_json,
                sharpe_ratio,
                win_rate,
                num_trades,
                max_drawdown,
                total_return,
                dsr,
                pbo,
                gate_passed,
                gate_reasons,
                report_path,
                datetime.now(),
            ],
        )
    logger.debug(
        "factory_run 保存: hash=%s [%s] sharpe=%s dsr=%s pbo=%s gate=%s",
        hypothesis_hash,
        market,
        sharpe_ratio,
        dsr,
        pbo,
        gate_passed,
    )


def load_factory_hashes() -> set[str]:
    """評価済み仮説ハッシュの集合を返す（サンプラーの重複試行防止用）。"""
    ensure_factory_tables()
    with _db_connection() as con:
        rows = con.execute("SELECT hypothesis_hash FROM factory_runs").fetchall()
    return {r[0] for r in rows}


def count_factory_runs() -> int:
    """累計評価数を返す（Deflated Sharpe の n_trials 補正に使用）。"""
    ensure_factory_tables()
    with _db_connection() as con:
        row = con.execute("SELECT COUNT(*) FROM factory_runs").fetchone()
    return int(row[0]) if row else 0
