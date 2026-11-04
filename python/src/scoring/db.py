"""採点用 PostgreSQL テーブルの作成と読み書き。

テーブル:
    rule_signal_outcomes … ルールのシグナルの採点結果
    rule_signal_versions … シグナルを記録した時点のアプリのバージョン
    prediction_history   … prediction_results の履歴（上書きされる前の写し）
    rule_best_history    … rule_best_by_symbol の週次の履歴
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.db.rule_results import ensure_rule_tables
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DDLS = (
    """
CREATE TABLE IF NOT EXISTS rule_signal_outcomes (
    signal_date   DATE      NOT NULL,
    market        VARCHAR   NOT NULL,
    symbol        VARCHAR   NOT NULL,
    rule_name     VARCHAR   NOT NULL,
    signal        INTEGER   NOT NULL,
    horizon       INTEGER   NOT NULL,
    hit           BOOLEAN,
    baseline      DOUBLE PRECISION,
    signed_return DOUBLE PRECISION,
    app_version   VARCHAR,
    scored_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_date, market, symbol, horizon)
)
""",
    """
CREATE TABLE IF NOT EXISTS rule_signal_versions (
    signal_date DATE      NOT NULL,
    market      VARCHAR   NOT NULL,
    symbol      VARCHAR   NOT NULL,
    app_version VARCHAR,
    logged_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (signal_date, market, symbol)
)
""",
    """
CREATE TABLE IF NOT EXISTS prediction_history (
    market           VARCHAR   NOT NULL,
    symbol           VARCHAR   NOT NULL,
    predicted_at     VARCHAR   NOT NULL,
    model_version    VARCHAR   NOT NULL,
    current_price    DOUBLE PRECISION,
    diff_ratio       DOUBLE PRECISION,
    diff_ratio_5d    DOUBLE PRECISION,
    app_version      VARCHAR,
    logged_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, symbol, predicted_at, model_version)
)
""",
    """
CREATE TABLE IF NOT EXISTS rule_best_history (
    evaluated_on DATE    NOT NULL,
    market       VARCHAR NOT NULL,
    symbol       VARCHAR NOT NULL,
    best_rule    VARCHAR NOT NULL,
    win_rate     DOUBLE PRECISION,
    num_trades   INTEGER,
    PRIMARY KEY (evaluated_on, market, symbol)
)
""",
)


def ensure_scoring_tables() -> None:
    with _db_connection() as con:
        for ddl in _DDLS:
            con.execute(ddl)


# ---------------------------------------------------------------------------
# 読み出し
# ---------------------------------------------------------------------------


def load_bars(market: str, start: date) -> pd.DataFrame:
    """日足（timeframe='1d'）の [symbol, ts, close, adj_close] を start 以降で返す。"""
    with _db_connection() as con:
        return pd.read_sql(
            """
            SELECT symbol, ts, close, adj_close FROM market_data_raw
            WHERE market = %s AND timeframe = '1d' AND ts >= %s
            ORDER BY symbol, ts
            """,
            con,
            params=[market, start],
        )


def load_rule_signals(market: str, start: date) -> pd.DataFrame:
    """rule_daily_signals に記録時のバージョンを左結合して返す。"""
    ensure_rule_tables()
    ensure_scoring_tables()
    with _db_connection() as con:
        df = pd.read_sql(
            """
            SELECT s.signal_date, s.market, s.symbol, s.rule_name, s.signal, s.price,
                   v.app_version
            FROM rule_daily_signals s
            LEFT JOIN rule_signal_versions v
              ON v.signal_date = s.signal_date AND v.market = s.market AND v.symbol = s.symbol
            WHERE s.market = %s AND s.signal_date >= %s
            ORDER BY s.signal_date, s.symbol
            """,
            con,
            params=[market, start],
        )
    # 版の無い昔の行は NaN でなく None で返す
    df["app_version"] = df["app_version"].astype(object).where(df["app_version"].notna(), None)
    return df


def load_prediction_signals_h1(market: str, start: date) -> pd.DataFrame:
    """prediction_accuracy の horizon=1 の行を返す（predicted_at は YYYYMMDD_HHMMSS の文字列）。"""
    with _db_connection() as con:
        return pd.read_sql(
            """
            SELECT symbol, model_name, predicted_at, predicted_ratio
            FROM prediction_accuracy
            WHERE market = %s AND horizon = 1 AND predicted_at >= %s
            ORDER BY predicted_at, symbol, model_name
            """,
            con,
            params=[market, start.strftime("%Y%m%d")],
        )


def load_prediction_history(market: str, start: date) -> pd.DataFrame:
    ensure_scoring_tables()
    with _db_connection() as con:
        return pd.read_sql(
            """
            SELECT * FROM prediction_history
            WHERE market = %s AND predicted_at >= %s
            ORDER BY predicted_at, symbol
            """,
            con,
            params=[market, start.strftime("%Y%m%d")],
        )


def load_rule_best(market: str) -> pd.DataFrame:
    ensure_rule_tables()
    with _db_connection() as con:
        return pd.read_sql(
            "SELECT * FROM rule_best_by_symbol WHERE market = %s ORDER BY symbol",
            con,
            params=[market],
        )


def load_rule_best_history(market: str) -> pd.DataFrame:
    ensure_scoring_tables()
    with _db_connection() as con:
        return pd.read_sql(
            "SELECT * FROM rule_best_history WHERE market = %s ORDER BY evaluated_on, symbol",
            con,
            params=[market],
        )


def load_rule_outcomes_scored(market: str, start: datetime, end: datetime) -> pd.DataFrame:
    """scored_at が [start, end) に入った採点の行を返す（時刻は UTC の naive）。"""
    ensure_scoring_tables()
    with _db_connection() as con:
        return pd.read_sql(
            """
            SELECT * FROM rule_signal_outcomes
            WHERE market = %s AND scored_at >= %s AND scored_at < %s
            ORDER BY signal_date, symbol, horizon
            """,
            con,
            params=[market, start, end],
        )


# ---------------------------------------------------------------------------
# 書き込み
# ---------------------------------------------------------------------------


def log_rule_signal_versions(signal_date: date, market: str, app_version: str | None) -> int:
    """その日のシグナル行に版を結びつける。足した行数を返す（既存行は触らない）。"""
    ensure_rule_tables()
    ensure_scoring_tables()
    with _db_connection() as con:
        cur = con.execute(
            """
            INSERT INTO rule_signal_versions (signal_date, market, symbol, app_version)
            SELECT signal_date, market, symbol, %s FROM rule_daily_signals
            WHERE signal_date = %s AND market = %s
            ON CONFLICT DO NOTHING
            """,
            [app_version, signal_date, market],
        )
        return max(cur.rowcount, 0)


def snapshot_predictions(app_version: str | None) -> int:
    """prediction_results の今の行を履歴へ写す。足した行数を返す。"""
    ensure_scoring_tables()
    with _db_connection() as con:
        cur = con.execute(
            """
            INSERT INTO prediction_history
                (market, symbol, predicted_at, model_version, current_price,
                 diff_ratio, diff_ratio_5d, app_version)
            SELECT market, symbol, predicted_at, model_version, current_price,
                   diff_ratio, diff_ratio_5d, %s
            FROM prediction_results
            ON CONFLICT DO NOTHING
            """,
            [app_version],
        )
        return max(cur.rowcount, 0)


def snapshot_rule_best(evaluated_on: date, market: str) -> int:
    """rule_best_by_symbol をその評価日の履歴へ写す（同じ日は上書き）。"""
    ensure_rule_tables()
    ensure_scoring_tables()
    with _db_connection() as con:
        cur = con.execute(
            """
            INSERT INTO rule_best_history
                (evaluated_on, market, symbol, best_rule, win_rate, num_trades)
            SELECT %s, market, symbol, best_rule, win_rate, num_trades
            FROM rule_best_by_symbol WHERE market = %s
            ON CONFLICT (evaluated_on, market, symbol) DO UPDATE SET
                best_rule = EXCLUDED.best_rule,
                win_rate = EXCLUDED.win_rate,
                num_trades = EXCLUDED.num_trades
            """,
            [evaluated_on, market],
        )
        return max(cur.rowcount, 0)


_OUTCOME_COLS = [
    "signal_date",
    "market",
    "symbol",
    "rule_name",
    "signal",
    "horizon",
    "hit",
    "baseline",
    "signed_return",
    "app_version",
    "scored_at",
]


def _py(v: Any) -> Any:
    """pandas/numpy の欠損と数値型を psycopg が受ける Python 値へ直す。"""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    item = getattr(v, "item", None)
    return item() if callable(item) else v


def _upsert_outcomes_sql() -> str:
    placeholders = ", ".join(["%s"] * len(_OUTCOME_COLS))
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in _OUTCOME_COLS
        if c not in ("signal_date", "market", "symbol", "horizon", "scored_at")
    )
    return (
        f"INSERT INTO rule_signal_outcomes ({', '.join(_OUTCOME_COLS)})"  # nosec B608
        f" VALUES ({placeholders})"
        f" ON CONFLICT (signal_date, market, symbol, horizon) DO UPDATE SET {updates}"
    )


def _outcome_values(df: pd.DataFrame) -> list[list[Any]]:
    return [[_py(r.get(c)) for c in _OUTCOME_COLS] for r in df.to_dict("records")]


def upsert_rule_outcomes(df: pd.DataFrame) -> int:
    """採点結果を主キーで上書き保存する。渡した行数を返す。

    scored_at は初めて採点した時刻として残し、上書きしない（週の集計が
    「その週に採点が増えた分」を scored_at で切るため）。
    """
    if df.empty:
        return 0
    ensure_scoring_tables()
    rows = _outcome_values(df)
    with _db_connection() as con:
        with con.cursor() as cur:
            cur.executemany(_upsert_outcomes_sql(), rows)
    return len(rows)


_DELETE_STALE_SQL = """
DELETE FROM rule_signal_outcomes o
WHERE o.market = %s AND o.signal_date >= %s AND o.signal_date <= %s
  AND NOT EXISTS (
    SELECT 1 FROM unnest(%s::date[], %s::text[], %s::int[]) AS k(d, s, h)
    WHERE k.d = o.signal_date AND k.s = o.symbol AND k.h = o.horizon
  )
"""


def replace_rule_outcomes(
    df: pd.DataFrame, market: str, since: date, until: date
) -> tuple[int, int]:
    """全期間の採点し直しの結果で、market の [since, until] の行を置き換える。

    1つのトランザクションで、主キーで上書き（scored_at は初めの値を残す）し、
    新しい結果に無い (signal_date, symbol, horizon) の行を消す。
    (書いた行数, 消した行数) を返す。新しい結果が空なら何もしない（読み込みの
    取りこぼしで全部を消さないため）。
    """
    if df.empty:
        return 0, 0
    ensure_scoring_tables()
    rows = _outcome_values(df)
    keys = df[df["market"] == market]
    with _db_connection() as con:
        with con.transaction():
            with con.cursor() as cur:
                cur.executemany(_upsert_outcomes_sql(), rows)
                cur.execute(
                    _DELETE_STALE_SQL,
                    [
                        market,
                        since,
                        until,
                        [_py(v) for v in keys["signal_date"]],
                        [str(v) for v in keys["symbol"]],
                        [int(v) for v in keys["horizon"]],
                    ],
                )
                deleted = cur.rowcount
    return len(rows), int(deleted)
