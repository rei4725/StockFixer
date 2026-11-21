"""
ルールベーストレーディング用 PostgreSQL テーブル操作

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

# paper_orders.side の買い（trading の OrderSide.BUY。utils からは trading を import できない）
_ORDER_SIDE_BUY = 1

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
    sell_only        BOOLEAN   NOT NULL DEFAULT FALSE,
    PRIMARY KEY (market, symbol)
)
"""

# 保有中だからと残した行の売りだけの印（#1797）。今ある表は migrations の 0009 でも足す
_DDL_BEST_SELL_ONLY = (
    "ALTER TABLE rule_best_by_symbol"
    " ADD COLUMN IF NOT EXISTS sell_only BOOLEAN NOT NULL DEFAULT FALSE"
)

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
        con.execute(_DDL_BEST_SELL_ONLY)
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
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    # 消すと書くを1つのトランザクションにし、INSERT が落ちたら前の行を残す（#2078）
    with _db_connection() as con, con.transaction():
        con.execute(
            "DELETE FROM rule_best_by_symbol WHERE market = %s AND symbol = %s",
            [market, symbol],
        )
        con.execute(
            """
            INSERT INTO rule_best_by_symbol
                (market, symbol, best_rule, win_rate, net_profit, num_trades,
                 profit_factor, max_drawdown, backtest_start, backtest_end, evaluated_at,
                 sell_only)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE)
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


def delete_rule_best(market: str, symbol: str) -> None:
    """1銘柄分の最優秀ルールの行を消す（評価が済み有効ルールが無かった銘柄用、#1275）。"""
    ensure_rule_tables()
    with _db_connection() as con:
        con.execute(
            "DELETE FROM rule_best_by_symbol WHERE market = %s AND symbol = %s",
            [market, symbol],
        )


def mark_rule_best_sell_only(market: str, symbol: str) -> None:
    """1銘柄分の行に売りだけの印を立てる（有効ルールが無いが保有中で残す行用、#1797）。

    印の立った行からは新規買いを出さず、保有の SELL だけを出す。
    有効ルールが選ばれて upsert_rule_best で書き直すと印は下りる。
    """
    ensure_rule_tables()
    with _db_connection() as con:
        con.execute(
            "UPDATE rule_best_by_symbol SET sell_only = TRUE WHERE market = %s AND symbol = %s",
            [market, symbol],
        )


def insert_rule_best_sell_only_if_missing(
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
) -> bool:
    """行が無いときだけ、売りだけの印を立てた行を足す（行の無い保有中の銘柄用、#1801）。

    行があれば何もしない（前のルールのまま。印は mark_rule_best_sell_only で立てる）。
    足したら True を返す。
    """
    ensure_rule_tables()
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with _db_connection() as con:
        cur = con.execute(
            """
            INSERT INTO rule_best_by_symbol
                (market, symbol, best_rule, win_rate, net_profit, num_trades,
                 profit_factor, max_drawdown, backtest_start, backtest_end, evaluated_at,
                 sell_only)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            ON CONFLICT (market, symbol) DO NOTHING
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
        return cur.rowcount > 0


def mark_rule_best_sell_only_except(market: str, keep_symbols: list[str]) -> None:
    """keep_symbols に無い銘柄の行に売りだけの印を立てる（ウォッチリストの外の行用、#1797）。

    keep_symbols が空なら何もしない（ウォッチリストの読み違いで全行に印を立てないため）。
    """
    if not keep_symbols:
        return
    ensure_rule_tables()
    placeholders = ", ".join(["%s"] * len(keep_symbols))
    with _db_connection() as con:
        con.execute(
            "UPDATE rule_best_by_symbol SET sell_only = TRUE"  # nosec B608 - only "%s"
            f" WHERE market = %s AND symbol NOT IN ({placeholders})",
            [market, *keep_symbols],
        )


def delete_rule_best_except(market: str, keep_symbols: list[str]) -> None:
    """keep_symbols に無い銘柄の最優秀ルールの行を消す（ウォッチリストから外れた銘柄用、#1294）。

    keep_symbols が空なら何も消さない（ウォッチリストの読み違いで全行を消さないため）。
    """
    if not keep_symbols:
        return
    ensure_rule_tables()
    placeholders = ", ".join(["%s"] * len(keep_symbols))
    with _db_connection() as con:
        con.execute(
            "DELETE FROM rule_best_by_symbol"  # nosec B608 - placeholders are only "%s"
            f" WHERE market = %s AND symbol NOT IN ({placeholders})",
            [market, *keep_symbols],
        )


def load_held_symbols() -> set[str]:
    """保有中（paper_positions の qty > 0）か、約定待ちの買いがある銘柄コードを返す（#1566）。

    週次評価が rule_best の行を消してよいかの確かめ用で、読めなければ例外を投げる。
    約定待ちの買いを足すのは、金曜の 16:30 の買いが約定する月曜より前に日曜の評価が走るため。
    空売りの建玉は足さない（ルールの SELL は買いの保有だけを見る）。
    """
    with _db_connection() as con:
        rows = con.execute(
            "SELECT symbol FROM paper_positions WHERE qty > 0"
            " UNION SELECT symbol FROM paper_orders WHERE status = 'pending' AND side = %s",
            [_ORDER_SIDE_BUY],
        ).fetchall()
    return {str(symbol).replace(".T", "") for (symbol,) in rows}


def load_open_position_symbols() -> set[str]:
    """load_held_symbols に空売りの建玉（paper_short_positions の qty > 0）を足して返す（#1615）。

    日次の取得と予想の対象を決める用で、読めなければ例外を投げる。
    """
    with _db_connection() as con:
        rows = con.execute("SELECT symbol FROM paper_short_positions WHERE qty > 0").fetchall()
    return load_held_symbols() | {str(symbol).replace(".T", "") for (symbol,) in rows}


def load_rule_best_all(market: str) -> pd.DataFrame:
    """有効な最優秀ルール一覧を返す（全銘柄）。"""
    ensure_rule_tables()
    with _db_connection() as con:
        return pd.read_sql(
            "SELECT * FROM rule_best_by_symbol WHERE market = %s ORDER BY win_rate DESC",
            con,
            params=[market],
        )


def load_effective_rules(
    market: str,
    min_win_rate: float = 0.5,
    min_net_profit: float = 0.0,
) -> pd.DataFrame:
    """判定基準を満たす銘柄と最優秀ルールに、売りだけの印の立った行を足して返す。

    売りだけの行は基準を外れていても返す（保有の SELL の判定に要る。#1801）。
    その行からは新規買いを出さないので、新規買いの基準は変わらない（#1797）。
    """
    ensure_rule_tables()
    with _db_connection() as con:
        return pd.read_sql(
            """
            SELECT * FROM rule_best_by_symbol
            WHERE market = %s
              AND (sell_only OR (win_rate >= %s AND net_profit > %s))
            ORDER BY win_rate DESC, net_profit DESC
            """,
            con,
            params=[market, min_win_rate, min_net_profit],
        )


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
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    # 消すと書くを1つのトランザクションにし、INSERT が落ちたら前の行を残す（#2078）
    with _db_connection() as con, con.transaction():
        con.execute(
            "DELETE FROM rule_daily_signals WHERE signal_date = %s AND market = %s AND symbol = %s",
            [str(signal_date), market, symbol],
        )
        con.execute(
            """
            INSERT INTO rule_daily_signals
                (signal_date, market, symbol, rule_name, signal, price, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [str(signal_date), market, symbol, rule_name, signal, price, now],
        )


def load_rule_signals_by_date(signal_date: date | str, market: str) -> pd.DataFrame:
    """指定日・マーケットのシグナル一覧を返す。"""
    ensure_rule_tables()
    with _db_connection() as con:
        return pd.read_sql(
            """
            SELECT * FROM rule_daily_signals
            WHERE signal_date = %s AND market = %s
            ORDER BY signal DESC, symbol
            """,
            con,
            params=[str(signal_date), market],
        )
