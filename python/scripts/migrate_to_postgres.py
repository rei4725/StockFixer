"""
DuckDB → PostgreSQL データ移行スクリプト（ビッグバング切り替え用、一回限り実行）

Usage:
    python scripts/migrate_to_postgres.py --duckdb-path data/stockfixer.duckdb

前提: 移行先PostgresにTask 2のマイグレーション（0001_baseline_postgres.sql等）が
適用済みであること。

冪等性: 各テーブルは INSERT 前に TRUNCATE するため、再実行しても安全。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# python/scripts/ から直接実行された場合（Usage欄の
# `python scripts/migrate_to_postgres.py` 形式）でも `src` を解決できるよう、
# python/ ルートをsys.pathへ追加する（run_*.py 系スクリプトと同じパターン）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402
import psycopg  # noqa: E402

from src.utils.data_path_utils import get_database_url  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

# テーブル名 → 明示カラムリスト（SELECT * による暗黙のカラム順一致に頼らない）
_TABLES: dict[str, list[str]] = {
    "stock_features": ["market", "symbol", "row_num"],  # 実際は動的列を持つため Step 2 で特別扱い
    "prediction_results": [
        "market",
        "symbol",
        "predicted_at",
        "model_version",
        "run_id",
        "current_price",
        "avg_pred_price",
        "diff_ratio",
        "model_count",
        "confidence_ratio",
        "avg_pred_price_3d",
        "avg_pred_price_5d",
        "avg_pred_price_10d",
        "diff_ratio_3d",
        "diff_ratio_5d",
        "diff_ratio_10d",
        "confluence_score",
    ],
    "market_data_raw": [
        "market",
        "symbol",
        "ticker",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "source",
        "ingested_at",
    ],
    "index_membership_history": [
        "market",
        "symbol",
        "index_name",
        "snapshot_date",
        "source",
        "fetched_at",
    ],
    "model_metrics": [
        "market",
        "symbol",
        "model_name",
        "trained_at",
        "rmse",
        "directional_accuracy",
        "n_samples",
    ],
    "prediction_accuracy": [
        "market",
        "symbol",
        "model_name",
        "predicted_at",
        "horizon",
        "predicted_price",
        "actual_price",
        "predicted_ratio",
        "actual_ratio",
        "direction_match",
        "checked_at",
    ],
    "paper_balance": ["balance"],
    "paper_orders": [
        "order_id",
        "market",
        "predicted_at",
        "symbol",
        "side",
        "qty",
        "price",
        "signal_price",
        "order_type",
        "status",
        "fill_price",
        "realized_pnl",
        "filled_at",
        "created_at",
        "horizon",
        "target_exit_date",
    ],
    "paper_positions": ["symbol", "qty", "avg_price", "updated_at"],
    "shap_values": [
        "market",
        "symbol",
        "model_name",
        "trained_at",
        "feature",
        "shap_mean",
        "shap_rank",
    ],
    "paper_real_diff": [
        "market",
        "symbol",
        "predicted_at",
        "side",
        "signal_price",
        "paper_order_id",
        "real_order_id",
        "paper_price",
        "real_price",
        "paper_slippage",
        "real_slippage",
        "price_diff",
        "paper_filled_at",
        "real_checked_at",
        "created_at",
        "updated_at",
        "order_session",
        "split_ratio",
    ],
    "feature_selection_log": [
        "market",
        "symbol",
        "model_name",
        "trained_at",
        "feature",
        "importance_mean",
        "importance_std",
        "importance_rank",
        "is_excluded",
        "protected_by_shap",
    ],
    "experiment_runs": [
        "run_id",
        "market",
        "symbol",
        "model_name",
        "trained_at",
        "horizon",
        "rmse",
        "directional_accuracy",
        "n_samples",
        "n_features",
        "feature_hash",
        "params_json",
        "created_at",
    ],
    "order_run_summary": [
        "run_id",
        "market",
        "mode",
        "run_at",
        "buy_orders",
        "sell_orders",
        "short_orders",
        "skipped",
        "skipped_min_change",
        "total_turnover",
        "min_change_ratio",
    ],
    "paper_short_positions": [
        "symbol",
        "qty",
        "avg_short_price",
        "unrealized_pnl",
        "opened_at",
        "updated_at",
    ],
    "dd_state": ["id", "peak_balance"],
    "data_quality_log": ["market", "symbol", "check_name", "level", "detail", "checked_at"],
    "system_config": ["key", "value", "updated_at"],
    "accuracy_weekly_snapshots": [
        "week_start",
        "market",
        "symbol",
        "direction_accuracy",
        "mean_abs_error",
        "n_samples",
        "snapshot_at",
    ],
    "earnings_calendar": ["market", "symbol", "event_date", "event_type", "fetched_at"],
    "stock_fundamentals": [
        "market",
        "symbol",
        "as_of",
        "revenue",
        "operating_income",
        "net_income",
        "eps",
        "roe",
        "op_margin",
        "net_margin",
        "debt_to_equity",
        "cash",
        "market_cap",
        "shares_outstanding",
        "revenue_cagr_3y",
        "fetched_at",
    ],
    "strategy_promotions": [
        "pr_number",
        "merge_commit_hash",
        "rule_or_feature_id",
        "promoted_at",
        "pre_promotion_baseline",
        "status",
    ],
    "factory_runs": [
        "hypothesis_hash",
        "market",
        "spec_json",
        "sharpe_ratio",
        "win_rate",
        "num_trades",
        "max_drawdown",
        "total_return",
        "dsr",
        "pbo",
        "gate_passed",
        "gate_reasons",
        "report_path",
        "evaluated_at",
    ],
    "claude_reasoning": ["run_id", "market", "thinking", "summary", "created_at"],
    "rule_best_by_symbol": [
        "market",
        "symbol",
        "best_rule",
        "win_rate",
        "net_profit",
        "num_trades",
        "profit_factor",
        "max_drawdown",
        "backtest_start",
        "backtest_end",
        "evaluated_at",
    ],
    "rule_daily_signals": [
        "signal_date",
        "market",
        "symbol",
        "rule_name",
        "signal",
        "price",
        "created_at",
    ],
}


def _get_dynamic_columns(src_con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """DuckDB側の実際のカラム一覧を取得する（stock_features等、動的にALTERされるテーブル用）

    注意: この関数は `main()` で `ATTACH ... AS pg (TYPE POSTGRES)` 実行後に
    呼ばれる。ATTACH後は DuckDB の `information_schema.columns` がローカルの
    移行元DBと、ATTACH先のPostgres（`pg`カタログ）の両方を横断するようになる。
    Postgres側にも移行先ベースラインスキーマ由来の同名テーブル（例:
    `stock_features` は market/symbol/row_num の3列）が存在するため、
    `table_catalog` で絞り込まないと両カタログ分のカラム名が重複して返り、
    後続の INSERT が `Duplicate column name` で失敗する。
    `current_database()` は ATTACH後も移行元DB名を指し続けるため、これで
    移行元DB自身のカラムだけに絞り込める。
    """
    rows = src_con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? AND table_catalog = current_database() "
        "ORDER BY ordinal_position",
        [table],
    ).fetchall()
    return [row[0] for row in rows]


def _get_dynamic_column_types(
    src_con: duckdb.DuckDBPyConnection, table: str
) -> list[tuple[str, str]]:
    """DuckDB側の実際のカラム名+型一覧を取得する（stock_features の不足カラムを
    Postgres側へ ALTER で補うための型推定に使う）。

    `_get_dynamic_columns()` と同様、`table_catalog = current_database()` で
    絞り込み、ATTACH先Postgresカタログ分の重複行を除外する。
    """
    rows = src_con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? AND table_catalog = current_database() "
        "ORDER BY ordinal_position",
        [table],
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _duckdb_type_to_postgres(duckdb_type: str) -> str:
    """DuckDBの `information_schema.columns.data_type` 文字列をPostgresの
    カラム型へマッピングする。

    `stock_features.py` の `_ensure_columns()`（pandas dtype起点の型推定）と
    同じ型カテゴリ（整数/浮動小数/真偽値/日時/文字列）に対応させる。
    未知の型（将来DuckDBが返しうる型を含む）は VARCHAR にフォールバックする
    （`_ensure_columns()` の else 節と同じ安全側の挙動）。
    """
    t = duckdb_type.upper()
    integer_types = {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
    }
    if t in integer_types:
        return "BIGINT"
    if t in ("DOUBLE", "FLOAT", "REAL") or t.startswith("DECIMAL"):
        return "DOUBLE PRECISION"
    if t == "BOOLEAN":
        return "BOOLEAN"
    if t.startswith("TIMESTAMP") or t == "DATE":
        return "TIMESTAMP"
    return "VARCHAR"


def _ensure_stock_features_target_columns(src_con: duckdb.DuckDBPyConnection) -> None:
    """`stock_features` について、DuckDB移行元にのみ存在する動的追加列
    （運用中に特徴量エンジニアリングがALTERしてきた列）をPostgres移行先へ補う。

    本スクリプトが動く時点（Postgresベースライン適用直後・アプリがまだ一度も
    Postgresへ書き込んでいない）では、Postgres側の `stock_features` は
    `0001_baseline_postgres.sql` 由来の market/symbol/row_num の3列しか
    持たない。一方、実運用のDuckDB移行元は `stock_features.py` の
    `_ensure_columns()` が運用中に `ALTER TABLE ADD COLUMN` してきた多数の
    指標列を持つ。この差分を先に埋めておかないと、後続の
    `INSERT INTO pg."stock_features" (...) SELECT ...` がPostgres側で
    「column does not exist」エラーで失敗する。

    注意1: ALTER は `pg.` カタログ経由（DuckDB接続からのDDLパススルー）ではなく、
    psycopg で直接Postgresへ実行する。実測したところ、DuckDBはALTER文中の
    型名を自分の型システムで正規化してからPostgresへ転送するため、
    `DOUBLE PRECISION`/`FLOAT8` はDuckDB内部エイリアスの `DOUBLE` に
    書き換えられて転送され、Postgres側で `type "double" does not exist`
    エラーになる（`stock_features` の指標列の大半は浮動小数点のため、
    これはstock_features全体で高確率に踏む問題であり、DuckDB経由のDDL
    パススルーには頼れない）。psycopgで直接Postgres構文のDDLを発行すれば
    この型名の再正規化を回避できる。

    注意2: DuckDBは一度ATTACHしたPostgresカタログのテーブルスキーマを接続の
    生存期間中キャッシュする。psycopgなど別接続経由でのスキーマ変更は、
    同じ `src_con` からは自動的に見えない（後続の `INSERT INTO pg."..."` が
    ALTER前の古いスキーマを見て「column does not exist」で失敗する）ことを
    実測で確認した。そのためカラムを追加した場合は `pg` カタログを
    DETACH→再ATTACHしてキャッシュを強制的に破棄する。
    """
    source_types = _get_dynamic_column_types(src_con, "stock_features")
    target_columns = {
        row[0]
        for row in src_con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'stock_features' AND table_catalog = 'pg'"
        ).fetchall()
    }
    missing = [(col, dt) for col, dt in source_types if col not in target_columns]
    if not missing:
        return

    with psycopg.connect(get_database_url(), autocommit=True) as pg_con:
        for col, duckdb_type in missing:
            pg_type = _duckdb_type_to_postgres(duckdb_type)
            pg_con.execute(f'ALTER TABLE stock_features ADD COLUMN IF NOT EXISTS "{col}" {pg_type}')
            logger.info(f"stock_features: Postgres側へカラムを追加しました: {col} ({pg_type})")

    # psycopg経由の変更をDuckDBの `pg` カタログキャッシュへ反映させるため、
    # DETACH→再ATTACHしてスキーマを再取得する。
    src_con.execute("DETACH pg")
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES, READ_ONLY FALSE)")


def migrate_table(src_con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> int:
    """1テーブル分をDuckDB→Postgresへ移行する。移行先件数を返す。

    冪等性: INSERT前にpg側をTRUNCATEするため、複数回実行しても最終状態は
    「DuckDB側の現在の内容」に収束する。

    `stock_features` は動的に列が増えるテーブルのため、TRUNCATE/INSERTの前に
    Postgres移行先の不足カラムをALTERで補う（他テーブルは静的スキーマで
    移行先とすでに一致している前提のため、この処理は行わない）。
    """
    if table == "stock_features":
        _ensure_stock_features_target_columns(src_con)

    col_list = ", ".join(f'"{c}"' for c in columns)
    src_con.execute(f'TRUNCATE TABLE pg."{table}"')
    src_con.execute(f'INSERT INTO pg."{table}" ({col_list}) SELECT {col_list} FROM "{table}"')
    row = src_con.execute(f'SELECT COUNT(*) FROM pg."{table}"').fetchone()
    count = int(row[0]) if row else 0
    logger.info(f"移行完了: {table} ({count}行)")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="DuckDB→PostgreSQL 一括データ移行")
    parser.add_argument("--duckdb-path", required=True, help="移行元DuckDBファイルパス")
    args = parser.parse_args()

    # 移行元DuckDBファイルは読み取り専用で開く（誤って書き換えないため）。
    # ただしDuckDBの read_only はデフォルトでATTACH先にも継承されるため、
    # 移行先Postgresへは明示的に READ_ONLY FALSE を指定して書き込みを許可する。
    src_con = duckdb.connect(args.duckdb_path, read_only=True)
    src_con.execute("INSTALL postgres")
    src_con.execute("LOAD postgres")
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES, READ_ONLY FALSE)")

    total = 0
    for table, columns in _TABLES.items():
        cols = _get_dynamic_columns(src_con, table) if table == "stock_features" else columns
        total += migrate_table(src_con, table, cols)

    logger.info(f"=== 移行完了: 全 {len(_TABLES)} テーブル、計 {total} 行 ===")
    src_con.close()


if __name__ == "__main__":
    main()
