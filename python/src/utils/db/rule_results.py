"""
ルールベーストレーディング用 DuckDB テーブル操作

テーブル:
    rule_best_by_symbol  … 銘柄ごとの最優秀ルール（週次評価で更新）
    rule_daily_signals   … 日次シグナル記録（日次ジョブで追記）
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_BEST_BY_SYMBOL = """
CREATE TABLE IF NOT EXISTS rule_best_by_symbol (
    market           VARCHAR   NOT NULL,
    symbol           VARCHAR   NOT NULL,
    best_rule        VARCHAR   NOT NULL,
    win_rate         DOUBLE    NOT NULL,
    net_profit       DOUBLE    NOT NULL,
    num_trades       INTEGER   NOT NULL,
    profit_factor    DOUBLE,
    max_drawdown     DOUBLE,
    backtest_start   DATE      NOT NULL,
    backtest_end     DATE      NOT NULL,
    evaluated_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, symbol)
)
"""

_DDL_DAILY_SIGNALS = """
CREATE TABLE IF NOT EXISTS rule_daily_signals (
    signal_date  DATE      NOT NULL,
    market       VARCHAR   NOT NULL,
    symbol       VARCHAR   NOT NULL,
    rule_name    VARCHAR   NOT NULL,
    signal       INTEGER   NOT NULL,
    price        DOUBLE,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_date, market, symbol)
)
"""


def ensure_rule_tables() -> None:
    with _db_connection() as con:
        con.execute(_DDL_BEST_BY_SYMBOL)
        con.execute(_DDL_DAILY_SIGNALS)


# ---------------------------------------------------------------------------
# rule_best_by_symbol
# ---------------------------------------------------------------------------


def upsert_rule_best(
    market: str,
    symbol: str,
    best_rule: str,
    win_rate: float,
    net_profit: float,
    num_trades: int,
    profit_factor: float | None,
    max_drawdown: float,
    backtest_start: str,
    backtest_end: str,
) -> None:
    ensure_rule_tables()
    from datetime import datetime

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with _db_connection() as con:
        con.execute(
            "DELETE FROM rule_best_by_symbol WHERE market = ? AND symbol = ?",
            [market, symbol],
        )
        con.execute(
            """
            INSERT INTO rule_best_by_symbol
                (market, symbol, best_rule, win_rate, net_profit, num_trades,
                 profit_factor, max_drawdown, backtest_start, backtest_end, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                market,
                symbol,
                best_rule,
                win_rate,
                net_profit,
                num_trades,
                profit_factor,
                max_drawdown,
                backtest_start,
                backtest_end,
                now,
            ],
        )


def load_rule_best_all(market: str) -> pd.DataFrame:
    """有効な最優秀ルール一覧を返す（全銘柄）。"""
    ensure_rule_tables()
    with _db_connection() as con:
        return con.execute(
            "SELECT * FROM rule_best_by_symbol WHERE market = ? ORDER BY win_rate DESC",
            [market],
        ).df()


def load_effective_rules(
    market: str,
    min_win_rate: float = 0.5,
    min_net_profit: float = 0.0,
) -> pd.DataFrame:
    """判定基準を満たす銘柄と最優秀ルールを返す。"""
    ensure_rule_tables()
    with _db_connection() as con:
        return con.execute(
            """
            SELECT * FROM rule_best_by_symbol
            WHERE market = ?
              AND win_rate  >= ?
              AND net_profit > ?
            ORDER BY win_rate DESC, net_profit DESC
            """,
            [market, min_win_rate, min_net_profit],
        ).df()


# ---------------------------------------------------------------------------
# rule_daily_signals
# ---------------------------------------------------------------------------


def upsert_rule_signal(
    signal_date: date | str,
    market: str,
    symbol: str,
    rule_name: str,
    signal: int,
    price: float | None,
) -> None:
    ensure_rule_tables()
    from datetime import datetime

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with _db_connection() as con:
        con.execute(
            "DELETE FROM rule_daily_signals WHERE signal_date = ? AND market = ? AND symbol = ?",
            [str(signal_date), market, symbol],
        )
        con.execute(
            """
            INSERT INTO rule_daily_signals
                (signal_date, market, symbol, rule_name, signal, price, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [str(signal_date), market, symbol, rule_name, signal, price, now],
        )


def load_rule_signals_by_date(signal_date: date | str, market: str) -> pd.DataFrame:
    """指定日・マーケットのシグナル一覧を返す。"""
    ensure_rule_tables()
    with _db_connection() as con:
        return con.execute(
            """
            SELECT * FROM rule_daily_signals
            WHERE signal_date = ? AND market = ?
            ORDER BY signal DESC, symbol
            """,
            [str(signal_date), market],
        ).df()
