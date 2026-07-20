# PostgreSQL移行 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** StockFixerのデータストアを埋め込み型DuckDBからPostgreSQL（自己ホスティング、docker-compose追加）へ完全移行し、`FileLock`直列化・テスト事故・スキーマ管理二重化を解消する。

**Architecture:** `docker-compose.yml`に`postgres`サービスを追加し、全プロセスがpsycopg3のコネクションプール経由で接続する。スキーマは`src/utils/db/migrations/`配下の番号付きSQLに一本化する。既存データはDuckDBのPostgres Attach拡張で一括移行（ビッグバング切り替え）する。テストはトランザクションロールバック方式に切り替える。

**Tech Stack:** PostgreSQL 16 (docker), psycopg3 (`psycopg[binary,pool]`), `psycopg_pool.ConnectionPool`, 既存のDuckDB（移行スクリプトでの読み出し専用）

## Global Constraints

- 対象範囲はPostgreSQL移行のみ。プロセス/サービス分割は別スペック（本計画では扱わない）。
- アクセス層はORMを導入せず、生SQL + psycopgを維持する（`docs/superpowers/specs/2026-07-19-postgres-migration-design.md`の方針）。
- 既存の`utils/db/`配下のモジュール構成・関数シグネチャ（呼び出し側から見える公開API）は変更しない。内部実装のみをpsycopgベースに置き換える。
- 各モジュール変換タスクは新規TDDではなく、既存のテストスイートを回帰ゲートとして使う（リファクタタスクのため）。新規コードにのみ新規テストを書く。
- SQL中のテーブル名・カラム名を文字列補間する箇所は、すべてハードコードされたリテラル値のみを対象とし、ユーザー入力を直接埋め込まない。

---

## Task 1: docker-compose への PostgreSQL サービス追加 + psycopg依存追加 + 接続先設定

**Files:**
- Modify: `docker-compose.yml`
- Modify: `python/requirements.txt`
- Modify: `python/src/utils/data_path_utils.py`
- Create: `python/.env.example`（存在しなければ新規、存在すれば追記）

**Interfaces:**
- Produces: `get_database_url() -> str`（`src/utils/data_path_utils.py`）。後続タスクの`_connection.py`が使用する。

- [ ] **Step 1: `docker-compose.yml` に postgres サービスを追加**

`docker-compose.yml`の`services:`ブロックに以下を追加する（`ollama`と`stockfixer`の間に挿入）:

```yaml
  postgres:
    image: postgres:16-alpine
    container_name: stockfixer-postgres
    restart: always
    environment:
      - POSTGRES_DB=stockfixer
      - POSTGRES_USER=stockfixer
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-stockfixer_dev}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U stockfixer -d stockfixer"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
```

`stockfixer`サービスの`depends_on`に`postgres`を追加し、`environment`に`DATABASE_URL`を追加する:

```yaml
    depends_on:
      ollama:
        condition: service_healthy
      postgres:
        condition: service_healthy
    environment:
      - LOG_FORMAT=json
      - OLLAMA_URL=http://ollama:11434
      - DATABASE_URL=postgresql://stockfixer:${POSTGRES_PASSWORD:-stockfixer_dev}@postgres:5432/stockfixer
```

末尾の`volumes:`ブロックに`postgres_data:`を追加する:

```yaml
volumes:
  ollama_data:
  postgres_data:
```

- [ ] **Step 2: requirements.txt に psycopg を追加**

`python/requirements.txt`の`duckdb==1.5.4`の次の行に追加する（DuckDBは移行スクリプト専用として残す）:

```
psycopg[binary,pool]>=3.2.3
```

- [ ] **Step 3: `get_database_url()` を data_path_utils.py に追加**

`python/src/utils/data_path_utils.py`の`# ===== DB関連パス =====`セクション（既存の`get_db_path()`の直後）に追加する:

```python
def get_database_url() -> str:
    """PostgreSQL接続文字列を返す（環境変数 DATABASE_URL 優先、未設定時はローカル既定値）"""
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://stockfixer:stockfixer_dev@localhost:5432/stockfixer",
    )
```

- [ ] **Step 4: `.env.example` に DATABASE_URL を追記**

`python/.env.example`が存在するか確認する:

```bash
ls python/.env.example 2>/dev/null || echo "not found"
```

存在すれば末尾に、存在しなければ新規作成して以下を書く:

```
DATABASE_URL=postgresql://stockfixer:stockfixer_dev@localhost:5432/stockfixer
POSTGRES_PASSWORD=stockfixer_dev
```

- [ ] **Step 5: ローカルで疎通確認**

```bash
docker compose up -d postgres
docker compose exec postgres pg_isready -U stockfixer -d stockfixer
```

Expected: `accepting connections`

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml python/requirements.txt python/src/utils/data_path_utils.py python/.env.example
git commit -m "feat: docker-composeにPostgreSQLサービスを追加しpsycopg依存を導入"
```

---

## Task 2: Postgresベースラインスキーマ + マイグレーションランナーのpsycopg移植

**Files:**
- Create: `python/src/utils/db/migrations/0001_baseline_postgres.sql`
- Create: `python/src/utils/db/migrations/0002_add_horizon_exit_date_postgres.sql`
- Modify: `python/src/utils/db/migration_runner.py`
- Test: `python/tests/unit/utils/db/test_migration_runner.py`

**Interfaces:**
- Consumes: なし（新規基盤）
- Produces: `run_migrations(con: psycopg.Connection, migrations_dir: str = _MIGRATIONS_DIR) -> int`、`get_applied_migrations(con) -> list[tuple[str, str, str]]`。Task 3の`_connection.py`が呼び出す。

現行`_connection.py`の`_init_tables()`（L142-484）には`migrations/0001_initial.sql`に未反映の4テーブル（`strategy_promotions`, `accuracy_weekly_snapshots`, `earnings_calendar`, `stock_fundamentals`）が存在する。Postgresへの移行を機にこの乖離を解消し、`_init_tables()`の現行内容全体を単一のPostgresベースラインとして書き起こす。

- [ ] **Step 1: `0001_baseline_postgres.sql` を作成**

`_init_tables()`（`_connection.py` L142-484）に定義された全17テーブル相当を、Postgres構文で書き起こす。型対応は `VARCHAR`→`VARCHAR`（そのまま）、`DOUBLE`→`DOUBLE PRECISION`、`BIGINT`/`INTEGER`/`BOOLEAN`/`TIMESTAMP`/`DATE`はそのまま。`DEFAULT CURRENT_TIMESTAMP`もそのまま使える。

```sql
-- 0001_baseline_postgres: 初期スキーマ全テーブル定義（Postgres版）
-- _connection.py の _init_tables() (2026-07-19時点) を正として書き起こし。
-- migrations/0001_initial.sql (DuckDB版) との既知の乖離
-- (strategy_promotions / accuracy_weekly_snapshots / earnings_calendar / stock_fundamentals
--  が反映されていなかった問題) をここで解消する。

CREATE TABLE IF NOT EXISTS stock_features (
    market   VARCHAR NOT NULL,
    symbol   VARCHAR NOT NULL,
    row_num  INTEGER NOT NULL,
    PRIMARY KEY (market, symbol, row_num)
);

CREATE TABLE IF NOT EXISTS prediction_results (
    market              VARCHAR NOT NULL,
    symbol              VARCHAR NOT NULL,
    predicted_at        VARCHAR NOT NULL,
    model_version       VARCHAR NOT NULL DEFAULT 'production',
    run_id              VARCHAR,
    current_price       DOUBLE PRECISION,
    avg_pred_price      DOUBLE PRECISION,
    diff_ratio          DOUBLE PRECISION,
    model_count         INTEGER,
    confidence_ratio    DOUBLE PRECISION,
    avg_pred_price_3d   DOUBLE PRECISION,
    avg_pred_price_5d   DOUBLE PRECISION,
    avg_pred_price_10d  DOUBLE PRECISION,
    diff_ratio_3d       DOUBLE PRECISION,
    diff_ratio_5d       DOUBLE PRECISION,
    diff_ratio_10d      DOUBLE PRECISION,
    confluence_score    INTEGER,
    PRIMARY KEY (market, symbol, predicted_at, model_version)
);

CREATE TABLE IF NOT EXISTS market_data_raw (
    market      VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    ticker      VARCHAR NOT NULL,
    timeframe   VARCHAR NOT NULL,
    ts          TIMESTAMP NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    adj_close   DOUBLE PRECISION,
    source      VARCHAR NOT NULL DEFAULT 'yfinance',
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, symbol, timeframe, ts)
);

CREATE TABLE IF NOT EXISTS index_membership_history (
    market        VARCHAR NOT NULL,
    symbol        VARCHAR NOT NULL,
    index_name    VARCHAR NOT NULL,
    snapshot_date DATE NOT NULL,
    source        VARCHAR NOT NULL DEFAULT 'wikipedia',
    fetched_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, symbol, snapshot_date)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    market               VARCHAR NOT NULL,
    symbol               VARCHAR NOT NULL,
    model_name           VARCHAR NOT NULL,
    trained_at           VARCHAR NOT NULL,
    rmse                 DOUBLE PRECISION,
    directional_accuracy DOUBLE PRECISION,
    n_samples            INTEGER,
    PRIMARY KEY (market, symbol, model_name, trained_at)
);

CREATE TABLE IF NOT EXISTS prediction_accuracy (
    market          VARCHAR NOT NULL,
    symbol          VARCHAR NOT NULL,
    model_name      VARCHAR NOT NULL,
    predicted_at    VARCHAR NOT NULL,
    horizon         INTEGER NOT NULL DEFAULT 1,
    predicted_price DOUBLE PRECISION,
    actual_price    DOUBLE PRECISION,
    predicted_ratio DOUBLE PRECISION,
    actual_ratio    DOUBLE PRECISION,
    direction_match BOOLEAN,
    checked_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, symbol, model_name, predicted_at, horizon)
);

CREATE TABLE IF NOT EXISTS paper_balance (
    balance DOUBLE PRECISION NOT NULL
);

INSERT INTO paper_balance (balance)
SELECT 1000000.0
WHERE NOT EXISTS (SELECT 1 FROM paper_balance);

CREATE TABLE IF NOT EXISTS paper_orders (
    order_id     VARCHAR NOT NULL PRIMARY KEY,
    market       VARCHAR,
    predicted_at VARCHAR,
    symbol       VARCHAR NOT NULL,
    side         INTEGER NOT NULL,
    qty          INTEGER NOT NULL,
    price        DOUBLE PRECISION,
    signal_price DOUBLE PRECISION,
    order_type   INTEGER NOT NULL,
    status       VARCHAR NOT NULL DEFAULT 'pending',
    fill_price   DOUBLE PRECISION,
    realized_pnl DOUBLE PRECISION,
    filled_at    TIMESTAMP,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    horizon      INTEGER,
    target_exit_date DATE
);

CREATE TABLE IF NOT EXISTS paper_positions (
    symbol      VARCHAR NOT NULL PRIMARY KEY,
    qty         INTEGER NOT NULL,
    avg_price   DOUBLE PRECISION NOT NULL,
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shap_values (
    market      VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    model_name  VARCHAR NOT NULL,
    trained_at  VARCHAR NOT NULL,
    feature     VARCHAR NOT NULL,
    shap_mean   DOUBLE PRECISION NOT NULL,
    shap_rank   INTEGER NOT NULL,
    PRIMARY KEY (market, symbol, model_name, trained_at, feature)
);

CREATE TABLE IF NOT EXISTS paper_real_diff (
    market          VARCHAR NOT NULL,
    symbol          VARCHAR NOT NULL,
    predicted_at    VARCHAR NOT NULL,
    side            INTEGER NOT NULL,
    signal_price    DOUBLE PRECISION,
    paper_order_id  VARCHAR,
    real_order_id   VARCHAR,
    paper_price     DOUBLE PRECISION,
    real_price      DOUBLE PRECISION,
    paper_slippage  DOUBLE PRECISION,
    real_slippage   DOUBLE PRECISION,
    price_diff      DOUBLE PRECISION,
    paper_filled_at TIMESTAMP,
    real_checked_at TIMESTAMP,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    order_session   VARCHAR,
    split_ratio     DOUBLE PRECISION,
    PRIMARY KEY (market, symbol, predicted_at, side)
);

CREATE TABLE IF NOT EXISTS feature_selection_log (
    market             VARCHAR NOT NULL,
    symbol             VARCHAR NOT NULL,
    model_name         VARCHAR NOT NULL,
    trained_at         VARCHAR NOT NULL,
    feature            VARCHAR NOT NULL,
    importance_mean    DOUBLE PRECISION NOT NULL,
    importance_std     DOUBLE PRECISION NOT NULL,
    importance_rank    INTEGER NOT NULL,
    is_excluded        BOOLEAN NOT NULL DEFAULT FALSE,
    protected_by_shap  BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (market, symbol, model_name, trained_at, feature)
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id               VARCHAR NOT NULL PRIMARY KEY,
    market               VARCHAR NOT NULL,
    symbol               VARCHAR NOT NULL,
    model_name           VARCHAR NOT NULL,
    trained_at           VARCHAR NOT NULL,
    horizon              INTEGER NOT NULL DEFAULT 1,
    rmse                 DOUBLE PRECISION,
    directional_accuracy DOUBLE PRECISION,
    n_samples            INTEGER,
    n_features           INTEGER,
    feature_hash         VARCHAR,
    params_json          VARCHAR,
    created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_run_summary (
    run_id             VARCHAR   NOT NULL PRIMARY KEY,
    market             VARCHAR   NOT NULL,
    mode               VARCHAR   NOT NULL,
    run_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    buy_orders         INTEGER   NOT NULL DEFAULT 0,
    sell_orders        INTEGER   NOT NULL DEFAULT 0,
    short_orders       INTEGER   NOT NULL DEFAULT 0,
    skipped            INTEGER   NOT NULL DEFAULT 0,
    skipped_min_change INTEGER   NOT NULL DEFAULT 0,
    total_turnover     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    min_change_ratio   DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS paper_short_positions (
    symbol          VARCHAR   NOT NULL PRIMARY KEY,
    qty             INTEGER   NOT NULL,
    avg_short_price DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION,
    opened_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dd_state (
    id           INTEGER PRIMARY KEY,
    peak_balance DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS data_quality_log (
    market      VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    check_name  VARCHAR NOT NULL,
    level       VARCHAR NOT NULL,
    detail      VARCHAR NOT NULL,
    checked_at  VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS system_config (
    key        VARCHAR NOT NULL PRIMARY KEY,
    value      VARCHAR NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS accuracy_weekly_snapshots (
    week_start         VARCHAR NOT NULL,
    market             VARCHAR NOT NULL,
    symbol             VARCHAR NOT NULL,
    direction_accuracy DOUBLE PRECISION,
    mean_abs_error     DOUBLE PRECISION,
    n_samples          INTEGER,
    snapshot_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (week_start, market, symbol)
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    market      VARCHAR   NOT NULL,
    symbol      VARCHAR   NOT NULL,
    event_date  DATE      NOT NULL,
    event_type  VARCHAR   NOT NULL DEFAULT 'earnings',
    fetched_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, symbol, event_date)
);

CREATE TABLE IF NOT EXISTS stock_fundamentals (
    market              VARCHAR   NOT NULL,
    symbol              VARCHAR   NOT NULL,
    as_of               TIMESTAMP NOT NULL,
    revenue             DOUBLE PRECISION,
    operating_income    DOUBLE PRECISION,
    net_income          DOUBLE PRECISION,
    eps                 DOUBLE PRECISION,
    roe                 DOUBLE PRECISION,
    op_margin           DOUBLE PRECISION,
    net_margin          DOUBLE PRECISION,
    debt_to_equity      DOUBLE PRECISION,
    cash                DOUBLE PRECISION,
    market_cap          DOUBLE PRECISION,
    shares_outstanding  DOUBLE PRECISION,
    revenue_cagr_3y     DOUBLE PRECISION,
    fetched_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, symbol)
);

-- 戦略ファクトリー昇格記録（旧 inline 定義のみ・migrations未反映だった）
CREATE TABLE IF NOT EXISTS strategy_promotions (
    pr_number              INTEGER   NOT NULL PRIMARY KEY,
    merge_commit_hash      VARCHAR   NOT NULL,
    rule_or_feature_id     VARCHAR   NOT NULL,
    promoted_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    pre_promotion_baseline VARCHAR,
    status                 VARCHAR   NOT NULL DEFAULT 'active'
);

-- 戦略ファクトリー評価結果（factory_runs.py が INSERT OR REPLACE する対象）
CREATE TABLE IF NOT EXISTS factory_runs (
    hypothesis_hash  VARCHAR NOT NULL PRIMARY KEY,
    market           VARCHAR NOT NULL,
    spec_json        VARCHAR,
    sharpe_ratio     DOUBLE PRECISION,
    win_rate         DOUBLE PRECISION,
    num_trades       INTEGER,
    max_drawdown     DOUBLE PRECISION,
    total_return     DOUBLE PRECISION,
    dsr              DOUBLE PRECISION,
    pbo              DOUBLE PRECISION,
    gate_passed      BOOLEAN,
    gate_reasons     VARCHAR,
    report_path      VARCHAR,
    evaluated_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Claude Extended Thinking 推論ログ（claude_agent.py が動的CREATEしていたもの）
CREATE TABLE IF NOT EXISTS claude_reasoning (
    run_id      VARCHAR PRIMARY KEY,
    market      VARCHAR,
    thinking    TEXT,
    summary     TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

- [ ] **Step 2: `0002_add_horizon_exit_date_postgres.sql` を作成**

Postgresは`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`をサポートするため、DuckDB版と同一の文で問題ない。ただしTask 2 Step 1のベースラインに既に`horizon`/`target_exit_date`を含めたため、このファイルは新規クラスタでは冪等にスキップされる（既存クラスタからの差分適用互換のためのみ残す）:

```sql
-- 0002_add_horizon_exit_date_postgres: paper_orders にホライズン情報と強制決済日を追加
-- (0001_baseline_postgres.sql に既に含まれているため、通常は no-op)
ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS horizon INTEGER;
ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS target_exit_date DATE;
```

- [ ] **Step 3: `migration_runner.py` をpsycopgベースに書き換え**

`python/src/utils/db/migration_runner.py`を全面置換する:

```python
"""
DB マイグレーションランナー（PostgreSQL / psycopg版）

src/utils/db/migrations/ ディレクトリの連番 SQL ファイルを
schema_migrations テーブルで管理しながら適用する。

命名規則:
  NNNN_description.sql          (フォワードマイグレーション)
  NNNN_description.rollback.sql (ロールバック用、*_postgres系は対象外)
"""

import os
import re
from typing import List, Tuple

import psycopg

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")
_VERSION_RE = re.compile(r"^(\d{4})_(.+?)_postgres\.sql$")


def _ensure_schema_migrations(con: psycopg.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     VARCHAR NOT NULL PRIMARY KEY,
            description VARCHAR NOT NULL,
            applied_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)


def _get_applied_versions(con: psycopg.Connection) -> set:
    rows = con.execute("SELECT version FROM schema_migrations").fetchall()
    return {str(row[0]) for row in rows}


def _discover_migrations(migrations_dir: str) -> List[Tuple[str, str, str]]:
    """
    フォワードマイグレーション SQL ファイル（*_postgres.sql）を
    (version, description, path) のリストで返す（昇順）。
    """
    if not os.path.isdir(migrations_dir):
        return []
    result = []
    for fname in sorted(os.listdir(migrations_dir)):
        m = _VERSION_RE.match(fname)
        if m:
            version = m.group(1)
            description = m.group(2).replace("_", " ")
            path = os.path.join(migrations_dir, fname)
            result.append((version, description, path))
    return result


def _split_statements(sql: str) -> List[str]:
    return [s.strip() for s in sql.split(";") if s.strip()]


def run_migrations(
    con: psycopg.Connection,
    migrations_dir: str = _MIGRATIONS_DIR,
) -> int:
    """未適用のマイグレーションを昇順に実行する。Returns: 適用したマイグレーション数"""
    _ensure_schema_migrations(con)
    applied = _get_applied_versions(con)
    pending = [
        (v, desc, path)
        for v, desc, path in _discover_migrations(migrations_dir)
        if v not in applied
    ]
    for version, description, path in pending:
        logger.info("マイグレーション適用: %s %s", version, description)
        with open(path, encoding="utf-8") as f:
            sql = f.read()
        for statement in _split_statements(sql):
            con.execute(statement)
        con.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (%s, %s)",
            [version, description],
        )
        logger.info("マイグレーション完了: %s", version)
    return len(pending)


def get_applied_migrations(con: psycopg.Connection) -> List[Tuple[str, str, str]]:
    _ensure_schema_migrations(con)
    rows = con.execute(
        "SELECT version, description, applied_at FROM schema_migrations ORDER BY version"
    ).fetchall()
    return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]
```

`_VERSION_RE`が`_postgres.sql`サフィックスのみを拾う点に注意（旧DuckDB版`0001_initial.sql`等は無視される）。

- [ ] **Step 4: テストを作成**

`python/tests/unit/utils/db/test_migration_runner.py`を新規作成する（このタスク時点ではまだTask 3のPostgres接続基盤がないため、`psycopg`の実接続なしでロジックのみ検証する）:

```python
import os

from src.utils.db.migration_runner import _discover_migrations, _split_statements


def test_discover_migrations_finds_postgres_suffixed_files(tmp_path):
    (tmp_path / "0001_baseline_postgres.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "0001_initial.sql").write_text("SELECT 1;", encoding="utf-8")  # DuckDB版は無視される
    (tmp_path / "0002_add_col_postgres.sql").write_text("SELECT 1;", encoding="utf-8")

    result = _discover_migrations(str(tmp_path))

    versions = [v for v, _, _ in result]
    assert versions == ["0001", "0002"]


def test_split_statements_ignores_blank_fragments():
    sql = "SELECT 1;\n\n  ;\nSELECT 2;"
    assert _split_statements(sql) == ["SELECT 1", "SELECT 2"]
```

- [ ] **Step 5: テスト実行**

```bash
cd python && python -m pytest tests/unit/utils/db/test_migration_runner.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/utils/db/migrations/0001_baseline_postgres.sql python/src/utils/db/migrations/0002_add_horizon_exit_date_postgres.sql python/src/utils/db/migration_runner.py python/tests/unit/utils/db/test_migration_runner.py
git commit -m "feat: Postgresベースラインスキーマとpsycopg版マイグレーションランナーを追加"
```

---

## Task 3: `_connection.py` をpsycopg3コネクションプールへ全面書き換え

**Files:**
- Modify: `python/src/utils/db/_connection.py`（全面書き換え）
- Modify: `python/src/utils/db/__init__.py`
- Test: `python/tests/unit/test_db_connection.py`（全面書き換え）

**Interfaces:**
- Consumes: `run_migrations(con)`（Task 2）, `get_database_url()`（Task 1）
- Produces:
  - `_db_connection(lock_timeout: float | None = None) -> ContextManager[psycopg.Connection]`
  - `set_test_connection(con: psycopg.Connection | None) -> None`（テスト専用フック、Task 4が使用）
  - `close_connection() -> None`
  - `get_readonly_connection() -> psycopg.Connection`
  - `DbLockTimeoutError(RuntimeError)`
  - `init_tables() -> None`

DuckDB版は各`con.execute(...)`が暗黙にautocommitされる前提で書かれており（呼び出し側に`.commit()`は一切ない）、この挙動を本番接続では`autocommit=True`で再現する。テスト時は`set_test_connection()`で注入された単一の接続（`autocommit=False`）を全呼び出しで共有し、明示`.commit()`が存在しないため何もコミットされず、フィクスチャ側の`rollback()`で完全に巻き戻せる。

- [ ] **Step 1: `_connection.py` を全面置換**

```python
"""
PostgreSQL 接続管理モジュール（psycopg3 + コネクションプール）

通常運用ではプロセス単位のコネクションプールから接続を借用する。
DuckDB版が暗黙のautocommitで動いていた（呼び出し側は一切 commit() しない）
挙動をそのまま踏襲するため、プール接続は autocommit=True で払い出す。

テスト時は set_test_connection() で単一の共有接続（autocommit=False）を
注入できる。呼び出し側が commit() しない設計のため、テスト終了時に
その接続を rollback() するだけで全ての変更を巻き戻せる。
"""

from contextlib import contextmanager
from typing import Generator, Optional

import psycopg
from psycopg_pool import ConnectionPool, PoolTimeout

from src.utils.data_path_utils import get_database_url
from src.utils.db.migration_runner import run_migrations
from src.utils.logger import get_logger

logger = get_logger(__name__)

_pool: Optional[ConnectionPool] = None
_tables_initialized = False
_test_connection: Optional[psycopg.Connection] = None

_DEFAULT_LOCK_TIMEOUT = 30.0


class DbLockTimeoutError(RuntimeError):
    """コネクションプールからの接続取得がタイムアウトしたことを表す。

    「別処理がDBを使用中で空き接続がない」ことを意味し、DB自体の異常ではない。
    ヘルスチェック等が busy と異常を区別するために使う。
    """


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            get_database_url(),
            min_size=1,
            max_size=4,
            kwargs={"autocommit": True},
            open=True,
        )
    return _pool


@contextmanager
def _db_connection(
    lock_timeout: float | None = None,
) -> Generator[psycopg.Connection, None, None]:
    """
    DB接続を提供するコンテキストマネージャー。

    Usage:
        with _db_connection() as con:
            df = con.execute("SELECT ...").fetchall()
    """
    global _tables_initialized

    if _test_connection is not None:
        yield _test_connection
        return

    timeout = _DEFAULT_LOCK_TIMEOUT if lock_timeout is None else lock_timeout
    try:
        with _get_pool().connection(timeout=timeout) as con:
            if not _tables_initialized:
                run_migrations(con)
                _tables_initialized = True
            yield con
    except PoolTimeout as e:
        raise DbLockTimeoutError(
            f"PostgreSQLコネクションプールからの接続取得がタイムアウトしました ({timeout}秒): {e}"
        ) from e


def set_test_connection(con: Optional[psycopg.Connection]) -> None:
    """テスト専用フック。以降の _db_connection() 呼び出しをこの接続に固定する。"""
    global _test_connection
    _test_connection = con


def close_connection() -> None:
    """状態リセット。プールを閉じ、テーブル初期化フラグを戻す（テスト間の切り替え用）"""
    global _tables_initialized, _pool
    _tables_initialized = False
    if _pool is not None:
        _pool.close()
        _pool = None


def get_readonly_connection() -> psycopg.Connection:
    """
    読み取り専用の新規接続を返す（呼び出し側で close() すること）。
    プールを介さない単発接続。
    """
    con = psycopg.connect(get_database_url(), autocommit=True)
    con.read_only = True
    return con


def init_tables() -> None:
    """外部から明示的にテーブル初期化する場合に使用"""
    with _db_connection() as con:
        run_migrations(con)
```

- [ ] **Step 2: `src/utils/db/__init__.py` のプロキシ転送対象を更新**

`_FORWARDED`フローズンセットから`get_db_path`を除去し、`get_database_url`を追加する。`get_data_dir`/`ensure_dir`は`_connection.py`がもう使わないため合わせて除去する:

```python
    _FORWARDED = frozenset(["_tables_initialized", "get_database_url"])
```

該当行（旧: `_FORWARDED = frozenset(["_tables_initialized", "get_db_path", "get_data_dir", "ensure_dir"])`）を上記に置換する。

同ファイル冒頭の`from src.utils.db._connection import (...)`ブロックに`get_database_url`のインポートは不要（`_connection.py`側でモジュールレベルインポートされていれば`_conn_module`経由で`__getattr__`が拾える。`_connection.py`のStep 1で`from src.utils.data_path_utils import get_database_url`をモジュールレベルで書いているため、この名前は`_connection`の名前空間に存在し、プロキシから参照可能）。

`from src.utils.db._connection import (...)`の対象リストから、存在しなくなった`_DB_CONFIG`, `_RETRY_DELAY`（新実装で削除したため）を除去する:

```python
from src.utils.db._connection import (
    _db_connection,
    close_connection,
    get_readonly_connection,
    init_tables,
    set_test_connection,
)
```

`from src.utils.db._connection import _RETRY_COUNT`の行は削除する（リトライ機構自体を廃止したため）。

- [ ] **Step 3: `test_db_connection.py` を全面書き換え**

旧ファイルはDuckDB接続とFileLockをモックしてリトライ・タイムアウト・ロック解放を検証していたが、新実装はプール方式でFileLockが存在しないため、テスト対象が変わる。`python/tests/unit/test_db_connection.py`を以下で全面置換する:

```python
"""_connection.py のユニットテスト（psycopg版）"""

from unittest.mock import MagicMock, patch

import pytest
from psycopg_pool import PoolTimeout

from src.utils.db._connection import (
    DbLockTimeoutError,
    _db_connection,
    close_connection,
    set_test_connection,
)


class TestDbConnectionTestMode:
    def teardown_method(self):
        set_test_connection(None)

    def test_uses_injected_test_connection_when_set(self):
        fake_con = MagicMock()
        set_test_connection(fake_con)

        with _db_connection() as con:
            assert con is fake_con

    def test_does_not_touch_pool_when_test_connection_set(self):
        fake_con = MagicMock()
        set_test_connection(fake_con)

        with patch("src.utils.db._connection._get_pool") as mock_get_pool:
            with _db_connection():
                pass
            mock_get_pool.assert_not_called()


class TestDbConnectionPoolTimeout:
    def teardown_method(self):
        close_connection()

    def test_raises_db_lock_timeout_error_on_pool_timeout(self):
        mock_pool = MagicMock()
        mock_pool.connection.side_effect = PoolTimeout("no connection available")

        with patch("src.utils.db._connection._get_pool", return_value=mock_pool):
            with pytest.raises(DbLockTimeoutError):
                with _db_connection(lock_timeout=0.1):
                    pass
```

- [ ] **Step 4: テスト実行**

```bash
cd python && python -m pytest tests/unit/test_db_connection.py -v
```

Expected: `3 passed`

（このタスク単体では実際のPostgres接続を伴うテストは含めない。実接続を要するテストはTask 4のfixture整備後に既存テスト群で検証される）

- [ ] **Step 5: Commit**

```bash
git add python/src/utils/db/_connection.py python/src/utils/db/__init__.py python/tests/unit/test_db_connection.py
git commit -m "feat: _connection.pyをpsycopg3コネクションプールへ全面書き換え"
```

---

## Task 4: テストフィクスチャのトランザクションロールバック方式への切り替え + 一括書き込みヘルパー新設

**Files:**
- Modify: `python/tests/unit/conftest.py`
- Modify: `python/tests/integration/conftest.py`
- Create: `python/src/utils/db/_bulk.py`
- Test: `python/tests/unit/utils/db/test_bulk.py`

**Interfaces:**
- Consumes: `set_test_connection()`, `close_connection()`（Task 3）
- Produces:
  - `bulk_insert(con: psycopg.Connection, table: str, df: pd.DataFrame, columns: list[str] | None = None) -> None`
  - `bulk_upsert(con: psycopg.Connection, table: str, df: pd.DataFrame, key_cols: list[str], columns: list[str] | None = None) -> None`
  - pytest fixture: `_isolate_db`（autouse, unit）— 各テストを1トランザクションに包み、テスト終了時に自動ロールバックする。以降の全モジュール変換タスクがこのfixtureに依存する。

- [ ] **Step 1: `_bulk.py` を新規作成**

DuckDBの`con.register(name, df)` + `INSERT ... SELECT * FROM name`に相当する処理をpsycopgの`COPY`で再現する:

```python
"""
DataFrame一括書き込みヘルパー（psycopg3版）

DuckDBの `con.register()` + `INSERT ... SELECT * FROM <df>` に相当する処理を
psycopg の COPY プロトコルで再現する。

table / columns / key_cols は必ずハードコードされたリテラル値を渡すこと
（ユーザー入力を直接埋め込まない）。
"""

from typing import Optional, Sequence

import pandas as pd
import psycopg


def _quoted_cols(cols: Sequence[str]) -> str:
    return ", ".join(f'"{c}"' for c in cols)


def _prepare_rows(df: pd.DataFrame, cols: Sequence[str]):
    clean = df[list(cols)].astype(object).where(pd.notnull(df[list(cols)]), None)
    return clean.itertuples(index=False, name=None)


def bulk_insert(
    con: psycopg.Connection,
    table: str,
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> None:
    """
    df の全行を table へ直接 COPY する（既存行との重複解決は行わない）。
    呼び出し側が事前に DELETE 済みのケース（stock_features 等）で使う。
    """
    if df.empty:
        return
    cols = list(columns) if columns is not None else list(df.columns)
    col_list = _quoted_cols(cols)

    with con.cursor() as cur:
        with cur.copy(f'COPY "{table}" ({col_list}) FROM STDIN') as copy:
            for row in _prepare_rows(df, cols):
                copy.write_row(row)


def bulk_upsert(
    con: psycopg.Connection,
    table: str,
    df: pd.DataFrame,
    key_cols: Sequence[str],
    columns: Optional[Sequence[str]] = None,
) -> None:
    """
    df の全行を table に upsert する（key_cols が既存行と一致すれば上書き）。
    DuckDB の `INSERT OR REPLACE` 相当。COPYで一時テーブルに流し込み、
    `INSERT ... ON CONFLICT (key_cols) DO UPDATE` で本テーブルへ反映する。
    """
    if df.empty:
        return
    cols = list(columns) if columns is not None else list(df.columns)
    col_list = _quoted_cols(cols)
    key_list = _quoted_cols(key_cols)
    update_cols = [c for c in cols if c not in key_cols]

    with con.cursor() as cur:
        cur.execute(
            f'CREATE TEMP TABLE _bulk_upsert ON COMMIT DROP AS '
            f'SELECT {col_list} FROM "{table}" WITH NO DATA'
        )
        with cur.copy(f"COPY _bulk_upsert ({col_list}) FROM STDIN") as copy:
            for row in _prepare_rows(df, cols):
                copy.write_row(row)

        if update_cols:
            set_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
            conflict_action = f"DO UPDATE SET {set_clause}"
        else:
            conflict_action = "DO NOTHING"

        cur.execute(
            f'INSERT INTO "{table}" ({col_list}) '
            f'SELECT {col_list} FROM _bulk_upsert '
            f'ON CONFLICT ({key_list}) {conflict_action}'
        )
```

- [ ] **Step 2: `_bulk.py` のテストを作成**

`ON COMMIT DROP`の一時テーブルはトランザクション内でのみ有効なため、テストは実Postgres接続を要する（Task 4 Step 3で整備するfixtureに依存）。このステップではfixtureがまだ無いため、テストファイルだけ作成し、実行はStep 4にまとめる:

```python
import pandas as pd

from src.utils.db._bulk import bulk_insert, bulk_upsert
from src.utils.db._connection import _db_connection


def test_bulk_insert_writes_all_rows():
    df = pd.DataFrame({"key": ["strategy-factory-idea"], "value": ["1"]})
    with _db_connection() as con:
        con.execute("CREATE TEMP TABLE _t_insert (key VARCHAR, value VARCHAR)")
        bulk_insert(con, "_t_insert", df)
        rows = con.execute("SELECT key, value FROM _t_insert").fetchall()
    assert rows == [("strategy-factory-idea", "1")]


def test_bulk_upsert_updates_existing_key():
    with _db_connection() as con:
        con.execute("CREATE TEMP TABLE _t_upsert (k VARCHAR PRIMARY KEY, v INTEGER)")
        con.execute("INSERT INTO _t_upsert VALUES ('a', 1)")
        df = pd.DataFrame({"k": ["a", "b"], "v": [99, 2]})
        bulk_upsert(con, "_t_upsert", df, key_cols=["k"])
        rows = dict(con.execute("SELECT k, v FROM _t_upsert ORDER BY k").fetchall())
    assert rows == {"a": 99, "b": 2}
```

- [ ] **Step 3: `tests/unit/conftest.py` の `_isolate_db` をトランザクションロールバック方式に置換**

`_isolate_db`と`_forbid_production_duckdb_connect`の2フィクスチャを、以下の1フィクスチャに置換する:

```python
@pytest.fixture(scope="session")
def _test_database_ready():
    """テストセッション開始時に1回だけ、テスト用Postgresへマイグレーションを適用する。

    DATABASE_URL は本番と共有の接続文字列だが、CI/ローカルとも
    テスト専用のPostgresインスタンス（docker-composeのpostgresサービス、
    または CI の services:postgres）を指す前提。
    """
    from src.utils.db._connection import _get_pool

    with _get_pool().connection() as con:
        from src.utils.db.migration_runner import run_migrations

        run_migrations(con)
    yield


@pytest.fixture(autouse=True)
def _isolate_db(_test_database_ready):
    """全 unit テストを1トランザクションに包み、テスト終了時にロールバックする。

    _connection.py は呼び出し側が commit() しない設計のため、共有接続を
    そのままロールバックするだけで全ての書き込みを巻き戻せる
    （#548: 本番DB破損事故 / PR#556: filelock起因のCI一斉失敗、両方の
    事故クラスがこの方式では構造的に発生しなくなる）。
    """
    import psycopg

    from src.utils.data_path_utils import get_database_url
    from src.utils.db._connection import close_connection, set_test_connection

    con = psycopg.connect(get_database_url(), autocommit=False)
    set_test_connection(con)
    try:
        yield
    finally:
        con.rollback()
        set_test_connection(None)
        con.close()
        close_connection()
```

`_block_discord_http`・`_block_heartbeat_ping`の2フィクスチャはそのまま残す（DB移行と無関係）。

- [ ] **Step 4: `tests/integration/conftest.py` にも同等のfixtureを追加**

`has_duckdb`フィクスチャの後に追加する（`has_duckdb`自体は移行スクリプトのテストで引き続き使うため残す）:

```python
@pytest.fixture(scope="session")
def _test_database_ready():
    from src.utils.db._connection import _get_pool
    from src.utils.db.migration_runner import run_migrations

    with _get_pool().connection() as con:
        run_migrations(con)
    yield


@pytest.fixture(autouse=True)
def _isolate_db(_test_database_ready):
    import psycopg

    from src.utils.data_path_utils import get_database_url
    from src.utils.db._connection import close_connection, set_test_connection

    con = psycopg.connect(get_database_url(), autocommit=False)
    set_test_connection(con)
    try:
        yield
    finally:
        con.rollback()
        set_test_connection(None)
        con.close()
        close_connection()
```

ファイル冒頭に`import pytest`しかない場合はそのままでよい（fixture内でローカルインポートしているため追加のトップレベルimportは不要）。

- [ ] **Step 5: ローカルPostgresを起動してテスト実行**

```bash
docker compose up -d postgres
cd python
$env:DATABASE_URL = "postgresql://stockfixer:stockfixer_dev@localhost:5432/stockfixer"
python -m pytest tests/unit/utils/db/test_bulk.py tests/unit/test_db_connection.py -v
```

Expected: 全件 `passed`

- [ ] **Step 6: Commit**

```bash
git add python/src/utils/db/_bulk.py python/tests/unit/utils/db/test_bulk.py python/tests/unit/conftest.py python/tests/integration/conftest.py
git commit -m "feat: テストDB隔離をトランザクションロールバック方式に切り替え、一括書き込みヘルパーを追加"
```

---

## Task 5: 縦切り移行 — `stock_features.py`

**Files:**
- Modify: `python/src/utils/db/stock_features.py`（全面書き換え）
- Test: 既存 `python/tests/unit/utils/db/test_stock_features.py`（存在すれば流用、無ければ最小限を新規作成）

**Interfaces:**
- Consumes: `bulk_insert()`（Task 4）, `_db_connection()`（Task 3）
- Produces: 既存公開API（`upsert_stock_features`, `load_stock_features`, `load_all_stock_features`, `delete_stock_features`, `get_all_symbols`, `_ensure_columns`）のシグネチャは変更しない。

- [ ] **Step 1: 既存テストの有無を確認**

```bash
ls python/tests/unit/utils/db/test_stock_features.py 2>/dev/null || echo "not found"
```

存在すればそのテストが以降のStep 4での回帰ゲートになる。存在しなければStep 1bで最小限のテストを新規作成する。

- [ ] **Step 1b (既存テストが無い場合のみ): 最小テストを新規作成**

```python
import pandas as pd

from src.utils.db.stock_features import (
    delete_stock_features,
    get_all_symbols,
    load_all_stock_features,
    load_stock_features,
    upsert_stock_features,
)


def test_upsert_and_load_roundtrip():
    df = pd.DataFrame({"close": [100.0, 101.0], "rsi": [50.0, 55.0]})
    upsert_stock_features("us", "TEST", df)

    loaded = load_stock_features("us", "TEST")

    assert loaded is not None
    assert list(loaded["close"]) == [100.0, 101.0]
    assert list(loaded["rsi"]) == [50.0, 55.0]


def test_upsert_adds_new_column_dynamically():
    df1 = pd.DataFrame({"close": [100.0]})
    upsert_stock_features("us", "TEST2", df1)

    df2 = pd.DataFrame({"close": [101.0], "new_indicator": [42.0]})
    upsert_stock_features("us", "TEST2", df2)

    loaded = load_stock_features("us", "TEST2")
    assert "new_indicator" in loaded.columns


def test_get_all_symbols_includes_saved_symbol():
    upsert_stock_features("jp", "9999", pd.DataFrame({"close": [1.0]}))
    assert ("jp", "9999") in get_all_symbols()


def test_delete_removes_data():
    upsert_stock_features("us", "TEST3", pd.DataFrame({"close": [1.0]}))
    delete_stock_features("us", "TEST3")
    assert load_stock_features("us", "TEST3") is None


def test_load_all_stock_features_combines_symbols():
    upsert_stock_features("us", "A1", pd.DataFrame({"close": [1.0]}))
    upsert_stock_features("us", "A2", pd.DataFrame({"close": [2.0]}))
    all_df = load_all_stock_features()
    assert set(all_df["symbol"]) >= {"A1", "A2"}
```

保存先: `python/tests/unit/utils/db/test_stock_features.py`

- [ ] **Step 2: 変更前のテストを実行して現状把握**

```bash
cd python && python -m pytest tests/unit/utils/db/test_stock_features.py -v
```

Expected: 全件 `FAIL`（`stock_features.py`がまだDuckDB版のままのため、Postgresに接続できずエラーになる）

- [ ] **Step 3: `stock_features.py` を全面書き換え**

```python
"""
stock_features テーブルの CRUD 操作

銘柄ごとのテクニカル指標・特徴量データを管理する。
"""

from typing import Optional

import pandas as pd
import psycopg

from src.utils.db._bulk import bulk_insert
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_columns(con: psycopg.Connection, df: pd.DataFrame) -> None:
    """DataFrame の列が stock_features テーブルに存在しない場合 ALTER TABLE で追加する。

    型推定を行い、適切な SQL 型で列を追加する。
    stock_features は特徴量エンジニアリングで列が頻繁に増えるため、
    migrations一本化の原則の例外として動的ALTERを維持する。
    """
    existing_cols: set = set()
    try:
        result = con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'stock_features'"
        ).fetchall()
        existing_cols = {row[0] for row in result}
    except Exception as e:
        logger.warning(f"stock_features カラム一覧取得失敗: {e}")

    reserved = {"market", "symbol", "row_num"}
    for col in df.columns:
        if col not in existing_cols and col not in reserved:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "BIGINT"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "DOUBLE PRECISION"
            elif pd.api.types.is_bool_dtype(dtype):
                sql_type = "BOOLEAN"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = "TIMESTAMP"
            else:
                sql_type = "VARCHAR"
            try:
                con.execute(f'ALTER TABLE stock_features ADD COLUMN IF NOT EXISTS "{col}" {sql_type}')
            except Exception:
                logger.debug(f"カラム追加スキップ（既存）: {col}")


def upsert_stock_features(market: str, symbol: str, df: pd.DataFrame) -> None:
    """
    指定 market/symbol の特徴量データを保存する。
    既存データは DELETE してから INSERT する（べき等）。
    """
    save_df = df.copy()
    if isinstance(save_df.index, pd.DatetimeIndex) and "date" not in save_df.columns:
        save_df = save_df.reset_index()
        first_col = save_df.columns[0]
        if first_col != "date":
            save_df = save_df.rename(columns={first_col: "date"})
    save_df["market"] = market
    save_df["symbol"] = symbol
    save_df["row_num"] = range(len(save_df))

    with _db_connection() as con:
        _ensure_columns(con, save_df)
        con.execute("DELETE FROM stock_features WHERE market = %s AND symbol = %s", [market, symbol])
        bulk_insert(con, "stock_features", save_df)
    logger.info(f"DB保存完了: stock_features [{market}_{symbol}] ({len(save_df)}行)")


def load_stock_features(market: str, symbol: str) -> Optional[pd.DataFrame]:
    """1銘柄分の特徴量を DB から取得する。Returns: 特徴量 DataFrame、データがなければ None"""
    with _db_connection() as con:
        try:
            df = pd.read_sql(
                "SELECT * FROM stock_features WHERE market = %(market)s AND symbol = %(symbol)s ORDER BY row_num",
                con,
                params={"market": market, "symbol": symbol},
            )
        except Exception as e:
            logger.error(f"stock_features 読み込み失敗 [{market}_{symbol}]: {e}", exc_info=True)
            return None

    if df.empty:
        return None

    drop_cols = [c for c in ["market", "symbol", "row_num"] if c in df.columns]
    return df.drop(columns=drop_cols)


def load_all_stock_features() -> pd.DataFrame:
    """全銘柄の特徴量を DB から取得する（統合モデル学習用）。"""
    with _db_connection() as con:
        try:
            df = pd.read_sql(
                "SELECT * FROM stock_features ORDER BY market, symbol, row_num", con
            )
        except Exception as e:
            logger.error(f"stock_features 全件読み込み失敗: {e}", exc_info=True)
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if "row_num" in df.columns:
        df = df.drop(columns=["row_num"])

    logger.info(f"DB読み込み完了: stock_features ({len(df)}行)")
    return df


def delete_stock_features(market: str, symbol: str) -> None:
    """指定 market/symbol のデータを削除する"""
    with _db_connection() as con:
        con.execute("DELETE FROM stock_features WHERE market = %s AND symbol = %s", [market, symbol])
    logger.info(f"DB削除完了: stock_features [{market}_{symbol}]")


def get_all_symbols() -> list:
    """stock_features テーブルに存在する全銘柄の (market, symbol) リストを返す。"""
    with _db_connection() as con:
        try:
            result = con.execute(
                "SELECT DISTINCT market, symbol FROM stock_features ORDER BY market, symbol"
            ).fetchall()
            return [(row[0], row[1]) for row in result]
        except Exception as e:
            logger.error(f"stock_features 銘柄一覧取得失敗: {e}", exc_info=True)
            return []
```

`pd.read_sql`はpsycopg3接続を渡すとSQLAlchemy非対応の警告（`UserWarning: pandas only supports SQLAlchemy connectable...`）を出すが、動作自体は正しい。この警告は許容する方針（設計書の「ORMを導入しない」方針を優先）。

- [ ] **Step 4: テスト再実行**

```bash
cd python && python -m pytest tests/unit/utils/db/test_stock_features.py -v
```

Expected: 全件 `PASS`

- [ ] **Step 5: Commit**

```bash
git add python/src/utils/db/stock_features.py python/tests/unit/utils/db/test_stock_features.py
git commit -m "feat: stock_features.pyをpsycopg3へ移行（縦切り検証）"
```

---

## Task 6: DataFrame一括upsert系モジュールの移行 — market_data.py / index_membership.py / event_calendar.py

**Files:**
- Modify: `python/src/utils/db/market_data.py`
- Modify: `python/src/utils/db/index_membership.py`
- Modify: `python/src/market_data/event_calendar.py`
- Test: 既存の対応するunitテストを回帰ゲートとして使用（新規テストは書かない）

**Interfaces:**
- Consumes: `bulk_upsert()`（Task 4）

3ファイルとも「`con.register(name, df)` → `INSERT OR REPLACE INTO tbl (...) SELECT ... FROM name`」という同一パターン。`bulk_upsert()`への置換で統一する。

- [ ] **Step 1: 対応する既存テストを確認**

```bash
cd python && python -m pytest tests/unit -k "market_data or index_membership or event_calendar" -v --collect-only
```

出力されたテストファイル群を、このタスクの回帰ゲートとして控えておく。

- [ ] **Step 2: `market_data.py` の書き込み部分を置換**

`upsert_raw_ohlcv`関数内（元L57-66付近）の

```python
    with _db_connection() as con:
        con.register("_raw_ohlcv_temp", df)
        con.execute("""
            INSERT OR REPLACE INTO market_data_raw
                (market, symbol, ticker, timeframe, ts,
                 open, high, low, close, volume, adj_close, source, ingested_at)
            SELECT market, symbol, ticker, timeframe, ts,
                   open, high, low, close, volume, adj_close, source, ingested_at
            FROM _raw_ohlcv_temp
        """)
```

相当の箇所を以下に置換する:

```python
    with _db_connection() as con:
        bulk_upsert(
            con,
            "market_data_raw",
            df,
            key_cols=["market", "symbol", "timeframe", "ts"],
            columns=[
                "market", "symbol", "ticker", "timeframe", "ts",
                "open", "high", "low", "close", "volume", "adj_close", "source", "ingested_at",
            ],
        )
```

ファイル冒頭に`from src.utils.db._bulk import bulk_upsert`を追加する。同ファイル内の他の`?`プレースホルダ（読み取りクエリ）は`%s`に、`.fetchdf()`は`pd.read_sql(sql, con, params=...)`に置換する（Task 5の`stock_features.py`と同じパターン）。

- [ ] **Step 3: `index_membership.py` の書き込み部分を置換**

元L74-81の

```python
    with _db_connection() as con:
        con.register("_index_membership_temp", df)
        con.execute("""
            INSERT OR REPLACE INTO index_membership_history
                (market, symbol, index_name, snapshot_date, source, fetched_at)
            SELECT market, symbol, index_name, snapshot_date, source, fetched_at
            FROM _index_membership_temp
        """)
```

を以下に置換する:

```python
    with _db_connection() as con:
        bulk_upsert(
            con,
            "index_membership_history",
            df,
            key_cols=["market", "symbol", "snapshot_date"],
            columns=["market", "symbol", "index_name", "snapshot_date", "source", "fetched_at"],
        )
```

ファイル冒頭に`from src.utils.db._bulk import bulk_upsert`を追加する。残る`?`プレースホルダ（1箇所）を`%s`に置換する。

- [ ] **Step 4: `event_calendar.py` の書き込み部分を置換**

元L186-192の

```python
            con.register("_ec_temp", df)
            con.execute("""
                INSERT OR REPLACE INTO earnings_calendar (market, symbol, event_date, event_type, fetched_at)
                SELECT market, symbol, event_date, event_type, CURRENT_TIMESTAMP
                FROM _ec_temp
            """)
```

を以下に置換する（`fetched_at`は`CURRENT_TIMESTAMP`をDataFrame側に付与してから渡す形に変える）:

```python
            import pandas as pd
            df = df.copy()
            df["fetched_at"] = pd.Timestamp.utcnow().tz_localize(None)
            bulk_upsert(
                con,
                "earnings_calendar",
                df,
                key_cols=["market", "symbol", "event_date"],
                columns=["market", "symbol", "event_date", "event_type", "fetched_at"],
            )
```

ファイル冒頭に`from src.utils.db._bulk import bulk_upsert`を追加する。`?`プレースホルダ（L83付近、`WHERE market = ? AND symbol = ?`）を`%s`に置換する。

- [ ] **Step 5: 回帰テスト実行**

```bash
cd python && python -m pytest tests/unit -k "market_data or index_membership or event_calendar" -v
```

Expected: Step 1で確認した全テストが `PASS`

- [ ] **Step 6: Commit**

```bash
git add python/src/utils/db/market_data.py python/src/utils/db/index_membership.py python/src/market_data/event_calendar.py
git commit -m "feat: market_data/index_membership/event_calendarをpsycopg3のbulk_upsertへ移行"
```

---

## Task 7: `prediction_results.py` の移行

**Files:**
- Modify: `python/src/prediction/db/prediction_results.py`

**Interfaces:**
- Consumes: `bulk_insert()`（Task 4。DELETE後の挿入のため upsert 不要）

- [ ] **Step 1: 既存テストを確認**

```bash
cd python && python -m pytest tests/unit -k "prediction_results" -v --collect-only
```

- [ ] **Step 2: 書き込み部分を置換**

元L71-79相当の

```python
        con.execute(
            "DELETE FROM prediction_results WHERE market = ? AND symbol = ? AND model_version = ?",
            [market, symbol, model_version],
        )
        ...
        col_str = ", ".join(cols)
        con.register("_save_df_temp", save_df)
        con.execute(
            f"INSERT INTO prediction_results ({col_str}) SELECT {col_str} FROM _save_df_temp"
        )
```

を以下に置換する:

```python
        con.execute(
            "DELETE FROM prediction_results WHERE market = %s AND symbol = %s AND model_version = %s",
            [market, symbol, model_version],
        )
        ...
        bulk_insert(con, "prediction_results", save_df, columns=cols)
```

ファイル冒頭に`from src.utils.db._bulk import bulk_insert`を追加する。

- [ ] **Step 3: 残る `?` プレースホルダを全て `%s` に置換**

ファイル内の全ての`?`（L125, 128, 136, 146, 150, 154, 158, 161, 193, 227, 232, 235, 238）を`%s`に置換する。動的にWHERE句を組み立てている箇所は、プレースホルダの個数と`params`リストの対応関係を崩さないよう注意する。

- [ ] **Step 4: `.fetchdf()` を `pd.read_sql` に置換**

L136, L168, L243の`con.execute(sql, params).fetchdf()`パターンを`pd.read_sql(sql, con, params=params)`に置換する（`params`がリストの場合はpsycopgの位置引数プレースホルダ`%s`のままリストを渡せる。辞書ベースにする必要はない）。

- [ ] **Step 5: 回帰テスト実行**

```bash
cd python && python -m pytest tests/unit -k "prediction_results" -v
```

Expected: Step 1で確認した全テストが `PASS`

- [ ] **Step 6: Commit**

```bash
git add python/src/prediction/db/prediction_results.py
git commit -m "feat: prediction_results.pyをpsycopg3へ移行"
```

---

## Task 8: 単一行upsert系モジュールの移行 — strategy_promotions.py / factory_runs.py / experiment.py / claude_agent.py

**Files:**
- Modify: `python/src/utils/db/strategy_promotions.py`
- Modify: `python/src/utils/db/factory_runs.py`
- Modify: `python/src/utils/db/experiment.py`
- Modify: `python/src/trading/claude_agent.py`

**Interfaces:**
- Consumes: なし（`INSERT ... ON CONFLICT`は生SQLで完結、`_bulk.py`不要）

4ファイルとも`INSERT OR REPLACE INTO tbl (cols) VALUES (?, ?, ...)`という単一行upsertパターン。`INSERT INTO tbl (cols) VALUES (%s, %s, ...) ON CONFLICT (主キー) DO UPDATE SET 非キー列 = EXCLUDED.非キー列`に置換する。

- [ ] **Step 1: `strategy_promotions.py` を置換**

元L53-58相当（主キー`pr_number`、列: `pr_number, merge_commit_hash, rule_or_feature_id, promoted_at, pre_promotion_baseline, status`）:

`save_strategy_promotion`関数には`status`という引数は存在しない（元コードでは`status`列は常にリテラル`'active'`で、`promoted_at`は関数引数（Noneなら`datetime.now()`で補完）としてバインドされている）。この2点を踏まえて以下に置換する:

```python
        con.execute(
            """
            INSERT INTO strategy_promotions
                (pr_number, merge_commit_hash, rule_or_feature_id, promoted_at, pre_promotion_baseline, status)
            VALUES (%s, %s, %s, %s, %s, 'active')
            ON CONFLICT (pr_number) DO UPDATE SET
                merge_commit_hash = EXCLUDED.merge_commit_hash,
                rule_or_feature_id = EXCLUDED.rule_or_feature_id,
                promoted_at = EXCLUDED.promoted_at,
                pre_promotion_baseline = EXCLUDED.pre_promotion_baseline,
                status = EXCLUDED.status
            """,
            [pr_number, merge_commit_hash, rule_or_feature_id, promoted_at, pre_promotion_baseline],
        )
```

（`promoted_at`は呼び出し元の関数内で`if promoted_at is None: promoted_at = datetime.now()`により補完済みの変数。ここをDB側の`CURRENT_TIMESTAMP`に置き換えてはならない — 呼び出し元が明示的に`promoted_at`を指定するケースを壊すため。）残る`?`（2箇所）を`%s`に、`.fetchdf()`（1箇所, L87）を`pd.read_sql`に置換する。

- [ ] **Step 2: `factory_runs.py` を置換**

主キー`hypothesis_hash`、列: `hypothesis_hash, market, spec_json, sharpe_ratio, win_rate, num_trades, max_drawdown, total_return, dsr, pbo, gate_passed, gate_reasons, report_path, evaluated_at`:

```python
        con.execute(
            """
            INSERT INTO factory_runs
                (hypothesis_hash, market, spec_json, sharpe_ratio, win_rate, num_trades,
                 max_drawdown, total_return, dsr, pbo, gate_passed, gate_reasons, report_path, evaluated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (hypothesis_hash) DO UPDATE SET
                market = EXCLUDED.market,
                spec_json = EXCLUDED.spec_json,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                win_rate = EXCLUDED.win_rate,
                num_trades = EXCLUDED.num_trades,
                max_drawdown = EXCLUDED.max_drawdown,
                total_return = EXCLUDED.total_return,
                dsr = EXCLUDED.dsr,
                pbo = EXCLUDED.pbo,
                gate_passed = EXCLUDED.gate_passed,
                gate_reasons = EXCLUDED.gate_reasons,
                report_path = EXCLUDED.report_path,
                evaluated_at = EXCLUDED.evaluated_at
            """,
            [hypothesis_hash, market, spec_json, sharpe_ratio, win_rate, num_trades,
             max_drawdown, total_return, dsr, pbo, gate_passed, gate_reasons, report_path, datetime.now()],
        )
```

元コードは`evaluated_at`を`CURRENT_TIMESTAMP`のようなDB側リテラルではなく、呼び出し時に`datetime.now()`をそのままパラメータとして渡している（関数引数ではなく呼び出し箇所でのインライン評価）。この挙動を変えないよう、`evaluated_at`用の14番目の`%s`にも必ず値を渡すこと。残る`?`（1箇所）を`%s`に置換する。

- [ ] **Step 3: `experiment.py` を置換**

主キー`run_id`、列: `run_id, market, symbol, model_name, trained_at, horizon, rmse, directional_accuracy, n_samples, n_features, feature_hash, params_json, created_at`:

```python
        con.execute(
            """
            INSERT INTO experiment_runs
                (run_id, market, symbol, model_name, trained_at, horizon,
                 rmse, directional_accuracy, n_samples, n_features, feature_hash, params_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                market = EXCLUDED.market,
                symbol = EXCLUDED.symbol,
                model_name = EXCLUDED.model_name,
                trained_at = EXCLUDED.trained_at,
                horizon = EXCLUDED.horizon,
                rmse = EXCLUDED.rmse,
                directional_accuracy = EXCLUDED.directional_accuracy,
                n_samples = EXCLUDED.n_samples,
                n_features = EXCLUDED.n_features,
                feature_hash = EXCLUDED.feature_hash,
                params_json = EXCLUDED.params_json
            """,
            [run_id, market, symbol, model_name, trained_at, horizon,
             rmse, directional_accuracy, n_samples, n_features, feature_hash, params_json, datetime.now()],
        )
```

元コードは`created_at`を`CURRENT_TIMESTAMP`のようなDB側リテラルではなく、呼び出し時に`datetime.now()`をそのままパラメータとして渡している。この挙動を変えないよう、`created_at`用の13番目の`%s`にも必ず値を渡すこと。残る`?`（4箇所）を`%s`に、`.fetchdf()`（2箇所, L135, L174）を`pd.read_sql`に置換する。

- [ ] **Step 4: `claude_agent.py` を置換**

元L375-390相当:

```python
        with _db_connection() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS claude_reasoning (
                    run_id      VARCHAR PRIMARY KEY,
                    market      VARCHAR,
                    thinking    TEXT,
                    summary     TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            con.execute(
                """
                INSERT INTO claude_reasoning (run_id, market, thinking, summary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    market = EXCLUDED.market,
                    thinking = EXCLUDED.thinking,
                    summary = EXCLUDED.summary
                """,
                [run_id, market, thinking_text, summary],
            )
```

`claude_reasoning`テーブルは`0001_baseline_postgres.sql`（Task 2）にも定義済みのため、この`CREATE TABLE IF NOT EXISTS`は冪等な保険として残す（削除しない）。

- [ ] **Step 5: 回帰テスト実行**

```bash
cd python && python -m pytest tests/unit -k "strategy_promotions or factory_runs or experiment or claude_agent" -v
```

Expected: 全件 `PASS`

- [ ] **Step 6: Commit**

```bash
git add python/src/utils/db/strategy_promotions.py python/src/utils/db/factory_runs.py python/src/utils/db/experiment.py python/src/trading/claude_agent.py
git commit -m "feat: strategy_promotions/factory_runs/experiment/claude_agentをON CONFLICT方式へ移行"
```

---

## Task 9: 精度・特徴量診断系モジュールの移行 — accuracy.py / model_metrics.py / features.py

**Files:**
- Modify: `python/src/prediction/db/accuracy.py`
- Modify: `python/src/prediction/db/model_metrics.py`
- Modify: `python/src/prediction/db/features.py`

- [ ] **Step 1: `accuracy.py` の `INSERT OR REPLACE` を置換**

主キー`(market, symbol, model_name, predicted_at, horizon)`、元L32-36:

```python
        con.execute(
            """
            INSERT INTO prediction_accuracy
                (market, symbol, model_name, predicted_at, horizon,
                 predicted_price, actual_price, predicted_ratio, actual_ratio, direction_match, checked_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (market, symbol, model_name, predicted_at, horizon) DO UPDATE SET
                predicted_price = EXCLUDED.predicted_price,
                actual_price = EXCLUDED.actual_price,
                predicted_ratio = EXCLUDED.predicted_ratio,
                actual_ratio = EXCLUDED.actual_ratio,
                direction_match = EXCLUDED.direction_match,
                checked_at = EXCLUDED.checked_at
            """,
            [market, symbol, model_name, predicted_at, horizon,
             predicted_price, actual_price, predicted_ratio, actual_ratio, direction_match],
        )
```

残る`?`（L79, 82, 85, 129, 132, 164, 203）を`%s`に、`.fetchdf()`（L93, 137, 178, 239）を`pd.read_sql`に置換する。

- [ ] **Step 2: `model_metrics.py` の `INSERT OR REPLACE` を置換**

主キー`(market, symbol, model_name, trained_at)`、元L30-32:

```python
        con.execute(
            """
            INSERT INTO model_metrics (market, symbol, model_name, trained_at, rmse, directional_accuracy, n_samples)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (market, symbol, model_name, trained_at) DO UPDATE SET
                rmse = EXCLUDED.rmse,
                directional_accuracy = EXCLUDED.directional_accuracy,
                n_samples = EXCLUDED.n_samples
            """,
            [market, symbol, model_name, trained_at, rmse, directional_accuracy, n_samples],
        )
```

残る`?`（L82, 109。`LIMIT ?`含む）を`%s`に置換する。

- [ ] **Step 3: `features.py`（shap_values / feature_selection_log）を置換**

このファイルは`INSERT OR REPLACE`を使わず、Delete→Insert方式（`DELETE`後に通常`INSERT`）のため、`?`→`%s`の置換のみでよい。L37, 73, 82, 87, 92, 97, 126, 151, 162, 182, 192の`?`を`%s`に、`.fetchdf()`（L87, 97）を`pd.read_sql`に置換する。

- [ ] **Step 4: 回帰テスト実行**

```bash
cd python && python -m pytest tests/unit -k "accuracy or model_metrics or features" -v
```

Expected: 全件 `PASS`

- [ ] **Step 5: Commit**

```bash
git add python/src/prediction/db/accuracy.py python/src/prediction/db/model_metrics.py python/src/prediction/db/features.py
git commit -m "feat: accuracy/model_metrics/featuresをpsycopg3へ移行"
```

---

## Task 10: 残る `utils/db/` モジュールの移行 — rule_results.py / stock_fundamentals.py / retention.py / quality_log.py / system_config.py / order_summary.py

**Files:**
- Modify: `python/src/utils/db/rule_results.py`
- Modify: `python/src/utils/db/stock_fundamentals.py`
- Modify: `python/src/utils/db/retention.py`
- Modify: `python/src/utils/db/quality_log.py`
- Modify: `python/src/utils/db/system_config.py`
- Modify: `python/src/prediction/db/order_summary.py`

このバッチは全て`?`プレースホルダの置換と`.fetchdf()`/`.df()`の置換のみで完結する（`INSERT OR REPLACE`を使っているファイルはこの中に無い）。

- [ ] **Step 1: `rule_results.py`**

`?`（9箇所）を`%s`に、`.df()`（L117, 137, 183）を`pd.read_sql`に置換する。

- [ ] **Step 2: `stock_fundamentals.py`**

`?`（3箇所）を`%s`に、`.fetchdf()`（L103, 122）を`pd.read_sql`に置換する。

- [ ] **Step 3: `retention.py`**

`?`（1箇所）を`%s`に置換する。

- [ ] **Step 4: `quality_log.py`**

`?`（1箇所）を`%s`に置換する。

- [ ] **Step 5: `system_config.py`**

`?`（2箇所）を`%s`に置換する。既存の`ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP`（L21-27）はPostgres/DuckDB両対応の構文のため、`ON CONFLICT`句自体は変更不要。ただし`excluded`は小文字表記だがPostgresでは大文字小文字を区別しない識別子として動作するためそのままでよい。

- [ ] **Step 6: `order_summary.py`**

`?`（L27-30の`VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)`）を`%s`に置換する（`INSERT OR REPLACE`ではなく通常`INSERT`のため、`ON CONFLICT`変換は不要）。

- [ ] **Step 7: 回帰テスト実行**

```bash
cd python && python -m pytest tests/unit -k "rule_results or stock_fundamentals or retention or quality_log or system_config or order_summary" -v
```

Expected: 全件 `PASS`

- [ ] **Step 8: Commit**

```bash
git add python/src/utils/db/rule_results.py python/src/utils/db/stock_fundamentals.py python/src/utils/db/retention.py python/src/utils/db/quality_log.py python/src/utils/db/system_config.py python/src/prediction/db/order_summary.py
git commit -m "feat: 残るutils/dbモジュール6本をpsycopg3へ移行"
```

---

## Task 11: 業務ロジック層の読み取り専用モジュールの移行 — paper_equity.py / correlation_risk.py / execution/predictions.py / drift_monitor.py / backtest/pipeline/features.py / backtest/slippage.py

**Files:**
- Modify: `python/src/trading/paper_equity.py`
- Modify: `python/src/trading/correlation_risk.py`
- Modify: `python/src/trading/execution/predictions.py`
- Modify: `python/src/prediction/drift_monitor.py`
- Modify: `python/src/backtest/pipeline/features.py`
- Modify: `python/src/backtest/slippage.py`

全て読み取り専用（`.fetchdf()`/`?`プレースホルダのみ、書き込みなし）。

- [ ] **Step 1: `paper_equity.py`**

L42-48の`.fetchdf()`を`pd.read_sql`に置換する（固定WHEREのため`?`プレースホルダなし、置換対象は`.fetchdf()`のみ）。

- [ ] **Step 2: `correlation_risk.py`**

L32の動的IN句プレースホルダ生成`placeholders = ", ".join("?" * len(symbols))`を`placeholders = ", ".join(["%s"] * len(symbols))`に置換する（`"%s" * n`は2文字文字列の反復ではなく文字単位の乱れた文字列になるバグを含むため、リストを`*`で複製してから`join`する必要がある）。L38・L44の`?`を`%s`に置換する。

- [ ] **Step 3: `execution/predictions.py`**

L23の`?`を`%s`に置換する。

- [ ] **Step 4: `drift_monitor.py`**

L59の`?`を`%s`に、L67の`.fetchdf()`を`pd.read_sql`に置換する。

- [ ] **Step 5: `backtest/pipeline/features.py`**

L158の`?`を`%s`に、L161の`.fetchdf()`を`pd.read_sql`に置換する。

- [ ] **Step 6: `backtest/slippage.py`**

`import duckdb`を削除し、`import psycopg`に置換する。`duckdb.DuckDBPyConnection`型ヒント（L66, L144）を`psycopg.Connection`に置換する。

- [ ] **Step 7: 回帰テスト実行**

```bash
cd python && python -m pytest tests/unit -k "paper_equity or correlation_risk or predictions or drift_monitor or slippage" -v
cd python && python -m pytest tests/unit/backtest/pipeline -v
```

Expected: 全件 `PASS`

- [ ] **Step 8: Commit**

```bash
git add python/src/trading/paper_equity.py python/src/trading/correlation_risk.py python/src/trading/execution/predictions.py python/src/prediction/drift_monitor.py python/src/backtest/pipeline/features.py python/src/backtest/slippage.py
git commit -m "feat: 業務ロジック層の読み取り専用DBアクセスをpsycopg3へ移行"
```

---

## Task 12: `compact.py` の物理コンパクション廃止 → Postgres VACUUM への置き換え

**Files:**
- Modify: `python/src/utils/db/compact.py`（DuckDB専用ロジックを削除し、VACUUM呼び出しに置換）
- Modify: `python/src/orchestration/jobs/weekly.py`
- Modify: `python/config/settings.py`（コメント更新のみ）

`compact.py`の`ATTACH`/`CHECKPOINT`/`swap_compacted`（ファイル入れ替え）はDuckDBの物理ファイル特有の概念で、Postgresには存在しない。Postgresでは`VACUUM (ANALYZE)`が対応する肥大化対策になる。

- [ ] **Step 1: `compact.py` を全面書き換え**

```python
"""
PostgreSQL メンテナンス（VACUUM）モジュール

DuckDB版は物理ファイルの再構築（ATTACH+コピー+ファイル入れ替え）を行っていたが、
Postgresでは VACUUM (ANALYZE) が対応する肥大化対策になる。
VACUUM はトランザクションブロック内では実行できないため、autocommit接続を使う。
"""

import psycopg

from src.utils.data_path_utils import get_database_url
from src.utils.logger import get_logger

logger = get_logger(__name__)


def vacuum_database() -> None:
    """DB全体に VACUUM (ANALYZE) を実行する。"""
    con = psycopg.connect(get_database_url(), autocommit=True)
    try:
        con.execute("VACUUM (ANALYZE)")
        logger.info("VACUUM (ANALYZE) 完了")
    finally:
        con.close()
```

- [ ] **Step 2: `weekly.py` の呼び出し箇所を更新**

`python/src/orchestration/jobs/weekly.py`のL341（`from src.utils.db.compact import compact_in_place`）を以下に置換する:

```python
from src.utils.db.compact import vacuum_database
```

L373-378相当（`compact_in_place(db_path, ...)`の呼び出しブロック）を以下に置換する:

```python
# 3. 月初週のみVACUUM（ANALYZE）で肥大化を回収する。
if DB_COMPACT_ENABLED and _is_first_week_of_month(now):
    logger.info("月初週のためVACUUM (ANALYZE) を実行します")
    vacuum_database()
    logger.info("VACUUM (ANALYZE) 完了")
```

`db_path = get_db_path()`の行（元L345）がこのブロック専用だった場合は削除する（他の箇所で`db_path`を使っていないか確認してから削除すること）。

- [ ] **Step 3: `settings.py` のコメントを更新**

`python/config/settings.py` L67-68のコメント（`VACUUM はファイルを縮小しないため、定期的な再構築で肥大の再発を防ぐ。`）を以下に更新する:

```python
# 月初週の週次メンテで VACUUM (ANALYZE) を実行し、肥大化・統計情報の陳腐化を防ぐか。
DB_COMPACT_ENABLED: bool = Field(default=True)
```

- [ ] **Step 4: 既存の compact 関連テストを削除・更新**

```bash
cd python && python -m pytest tests/unit -k "compact" -v --collect-only
```

出力されたテストファイル（DuckDBのATTACH/CHECKPOINT挙動をテストしていたもの）を確認し、`vacuum_database()`をテストする内容に置き換える:

```python
from unittest.mock import MagicMock, patch

from src.utils.db.compact import vacuum_database


def test_vacuum_database_executes_vacuum_analyze():
    mock_con = MagicMock()
    with patch("src.utils.db.compact.psycopg.connect", return_value=mock_con):
        vacuum_database()
    mock_con.execute.assert_called_once_with("VACUUM (ANALYZE)")
    mock_con.close.assert_called_once()
```

- [ ] **Step 5: テスト実行**

```bash
cd python && python -m pytest tests/unit -k "compact" -v
```

Expected: 全件 `PASS`

- [ ] **Step 6: Commit**

```bash
git add python/src/utils/db/compact.py python/src/orchestration/jobs/weekly.py python/config/settings.py python/tests/unit/utils/db/test_compact.py
git commit -m "feat: DuckDB物理コンパクションをPostgres VACUUM (ANALYZE)へ置き換え"
```

---

## Task 12.5: `backup_pipeline.py` を pg_dump ベースの日次バックアップへ移行

**Files:**
- Modify: `python/src/orchestration/backup_pipeline.py`（全面書き換え）
- Test: 既存の対応するunitテストを確認し、更新する

**Interfaces:**
- Consumes: `get_database_url()`（Task 1）
- Produces: `run_db_backup() -> dict`（既存シグネチャ維持。呼び出し元 `python/src/orchestration/jobs/daily.py` の `result["backup_path"]`/`result["size_mb"]`/`result["elapsed_seconds"]`/`result["pruned_count"]`/`result["error"]` 参照はそのまま動作させる）

Task 12で `weekly.py` のDuckDB専用 `CHECKPOINT` を除去したのと同じ理由で、`backup_pipeline.py` の `run_db_backup()` も修正が必要。現状は `CHECKPOINT` 実行後にDuckDBファイルを直接コピーする方式だが、Postgres移行後は (1) `CHECKPOINT` がPostgresでは一般ユーザー権限で実行できず失敗し、(2) データがコンテナ内のPostgresボリュームに存在するためファイルコピーでは何も意味のあるバックアップにならない。`docker compose exec postgres which pg_dump pg_restore` で確認済みの通り、`postgres:16-alpine` イメージには `pg_dump`/`pg_restore` が同梱されている。Pythonの `subprocess` から `pg_dump` をカスタムフォーマット（`-Fc`）で呼び出し、単一ファイルのバックアップとして保存する方式に置き換える。

- [ ] **Step 1: 既存テストの有無を確認**

```bash
cd python && python -m pytest tests/unit -k "backup_pipeline" -v --collect-only
```

出力されたテストを、このタスクの回帰ゲートとして控えておく。

- [ ] **Step 2: `backup_pipeline.py` を全面書き換え**

```python
"""
PostgreSQL 定期バックアップパイプライン (NF-602)

pg_dump（カスタムフォーマット）でタイムスタンプ付きファイルへ出力し、最大5世代を保持する。
"""

import os
import shutil
import subprocess
import time
from datetime import datetime
from urllib.parse import urlparse

from src.utils.data_path_utils import get_data_dir, get_database_url
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_GENERATIONS = 5
_PG_DUMP_TIMEOUT_SECONDS = 300


def get_backup_dir() -> str:
    """バックアップルートディレクトリのパスを返す"""
    return os.path.join(get_data_dir(), "backups")


def run_db_backup() -> dict:
    """
    PostgreSQL バックアップを実行する。

    手順:
        1. pg_dump（カスタムフォーマット, -Fc）でタイムスタンプ付きファイルへ出力
        2. 5世代超過分を古い順に削除

    Returns:
        dict: backup_path, size_mb, elapsed_seconds, pruned_count, error
    """
    backup_root = get_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dest_dir = os.path.join(backup_root, timestamp)
    backup_dest_path = os.path.join(backup_dest_dir, "stockfixer.dump")

    start = time.monotonic()
    error_msg = None
    size_mb = 0.0
    pruned_count = 0

    try:
        os.makedirs(backup_dest_dir, exist_ok=True)
        logger.info("バックアップ: pg_dump 開始 → %s", backup_dest_path)
        _run_pg_dump(get_database_url(), backup_dest_path)
        size_mb = os.path.getsize(backup_dest_path) / (1024 * 1024)
        logger.info("バックアップ: pg_dump 完了 (%.2f MB)", size_mb)

        pruned_count = _prune_old_backups(backup_root, MAX_GENERATIONS)

    except Exception as e:
        logger.error("バックアップ失敗: %s", e, exc_info=True)
        error_msg = str(e)

    elapsed = time.monotonic() - start
    logger.info(
        "=== バックアップ完了: %.1f 秒, %.2f MB, 削除世代=%s ===",
        elapsed,
        size_mb,
        pruned_count,
    )

    return {
        "backup_path": backup_dest_path,
        "size_mb": size_mb,
        "elapsed_seconds": elapsed,
        "pruned_count": pruned_count,
        "error": error_msg,
    }


def _run_pg_dump(database_url: str, dest_path: str) -> None:
    """pg_dump をカスタムフォーマットで実行する（環境変数でパスワードを渡し、コマンドラインへの露出を避ける）。"""
    parsed = urlparse(database_url)
    env = dict(os.environ)
    if parsed.password:
        env["PGPASSWORD"] = parsed.password

    cmd = [
        "pg_dump",
        "-Fc",
        "-h",
        parsed.hostname or "localhost",
        "-p",
        str(parsed.port or 5432),
        "-U",
        parsed.username or "",
        "-f",
        dest_path,
        (parsed.path or "/").lstrip("/"),
    ]
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=_PG_DUMP_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pg_dump 失敗 (code={result.returncode}): {result.stderr}")


def _prune_old_backups(backup_root: str, max_generations: int) -> int:
    """max_generations を超えた古いバックアップを削除し、削除件数を返す。"""
    if not os.path.isdir(backup_root):
        return 0

    # YYYYMMDD_HHMMSS 形式のディレクトリのみ対象（辞書順 = 時系列順）
    entries = sorted(
        e for e in os.listdir(backup_root) if os.path.isdir(os.path.join(backup_root, e))
    )
    to_delete = entries[: max(0, len(entries) - max_generations)]

    for name in to_delete:
        target = os.path.join(backup_root, name)
        try:
            shutil.rmtree(target)
            logger.info("古いバックアップを削除: %s", target)
        except Exception as e:
            logger.error("バックアップ削除失敗 (%s): %s", target, e, exc_info=True)

    return len(to_delete)
```

`_prune_old_backups` はDBエンジンに依存しないロジックのため元コードのまま維持する。`get_backup_dir()` の公開シグネチャも維持する。

- [ ] **Step 3: `pg_dump` がコンテナ内で利用可能か確認**

`docker-compose.yml` の `stockfixer` サービスは `python/Dockerfile` からビルドされる。このイメージに `pg_dump`（PostgreSQLクライアントツール）が含まれているか確認する:

```bash
docker compose run --rm stockfixer which pg_dump
```

含まれていない場合は `python/Dockerfile` に `postgresql-client` パッケージのインストールを追加する（Debian系ベースイメージなら `apt-get install -y postgresql-client`、Alpine系なら `apk add postgresql-client` — 実際のベースイメージに応じた行を追加すること）。

- [ ] **Step 4: 既存テストを更新**

Step 1で見つけたテストファイルを読み、DuckDBの `CHECKPOINT`/`shutil.copy2` を前提にしたモックを、新しい `_run_pg_dump`（`subprocess.run` をモック）を前提にした形に書き換える。`_prune_old_backups` を検証するテストはロジック変更が無いためそのまま流用できる。

- [ ] **Step 5: テスト実行**

```bash
cd python && python -m pytest tests/unit -k "backup_pipeline" -v
```

Expected: 全件 `PASS`

- [ ] **Step 6: ローカルで実際にバックアップを実行して検証**

```bash
docker compose up -d postgres
cd python && python -c "from src.orchestration.backup_pipeline import run_db_backup; print(run_db_backup())"
```

Expected: `error` キーが `None`、`backup_path` に指定した `.dump` ファイルが実在し、`pg_restore --list <path>` でダンプ内容が読めること。

- [ ] **Step 7: Commit**

```bash
git add python/src/orchestration/backup_pipeline.py python/Dockerfile
git commit -m "feat: backup_pipeline.pyをpg_dumpベースのPostgresバックアップへ移行"
```

（`python/Dockerfile` は Step 3 で変更が必要だった場合のみ追加。テストファイルも変更していれば併せて追加すること。）

---

## Task 13: `health.py` のDBヘルスチェックをpsycopg版へ移行

**Files:**
- Modify: `python/src/api/health.py`
- Modify: `python/tests/unit/test_health_endpoint.py`

**Interfaces:**
- Consumes: `_db_connection()`, `DbLockTimeoutError`（Task 3）

- [ ] **Step 1: `_check_db()` を更新**

`python/src/api/health.py`のL59（`from src.utils.db._connection import DbLockTimeoutError, _db_connection`）はインポート元がそのまま使えるため変更不要。`_check_db()`本体（L41-69）のロジックも、`_db_connection(lock_timeout=_DB_CHECK_LOCK_TIMEOUT)`のシグネチャが維持されているため変更不要。`con.execute("SELECT 1").fetchone()`もpsycopgでそのまま動作する。

L47のコメント（`FileLock 直列化・設定統一`）のみ実態に合わせて更新する:

```python
    # コネクションプール経由で読む（プールが空の場合は DbLockTimeoutError で busy 扱いにする）。
```

- [ ] **Step 2: `test_health_endpoint.py` を確認・更新**

`TestDbConnectionLockTimeout.test_raises_db_lock_timeout_error_when_locked`（L206-221）は`FileLock`を直接インスタンス化してロック競合を模擬していたが、この仕組みは廃止された。以下に置換する:

```python
class TestDbConnectionLockTimeout:
    def test_raises_db_lock_timeout_error_when_pool_exhausted(self):
        from unittest.mock import MagicMock, patch

        from psycopg_pool import PoolTimeout

        from src.utils.db._connection import DbLockTimeoutError, _db_connection

        mock_pool = MagicMock()
        mock_pool.connection.side_effect = PoolTimeout("exhausted")

        with patch("src.utils.db._connection._get_pool", return_value=mock_pool):
            with pytest.raises(DbLockTimeoutError):
                with _db_connection(lock_timeout=0.1):
                    pass
```

`TestCheckDb.test_ok_when_db_available`・`test_single_connection_returns_last_prediction`（L165-191）は`_isolate_db`autouse fixture（Task 4で刷新済み）に依存する実DB接続テストのため、コード変更は不要（Task 4のfixture刷新により自動的にPostgres経路で動く）。

- [ ] **Step 3: テスト実行**

```bash
cd python && python -m pytest tests/unit/test_health_endpoint.py -v
```

Expected: 全件 `PASS`

- [ ] **Step 4: Commit**

```bash
git add python/src/api/health.py python/tests/unit/test_health_endpoint.py
git commit -m "feat: health.pyのDBヘルスチェックをpsycopgプール前提のコメント・テストに更新"
```

---

## Task 14: データ移行スクリプト `scripts/migrate_to_postgres.py`

**Files:**
- Create: `python/scripts/migrate_to_postgres.py`
- Test: `python/tests/integration/scripts/test_migrate_to_postgres.py`

このスクリプトはDuckDBのPostgres Attach拡張を使い、既存の全テーブルをPostgresへ一括移行する。カラム順の暗黙一致に頼る`SELECT *`は使わず、明示的なカラムリストを使う（`0001_baseline_postgres.sql`のテーブル定義がDuckDB側のALTER履歴により列追加順が異なる可能性があるため）。

- [ ] **Step 1: 移行対象テーブルとカラムリストを定義**

```python
"""
DuckDB → PostgreSQL データ移行スクリプト（ビッグバング切り替え用、一回限り実行）

Usage:
    python scripts/migrate_to_postgres.py --duckdb-path data/stockfixer.duckdb

前提: 移行先PostgresにTask 2のマイグレーション（0001_baseline_postgres.sql等）が
適用済みであること。

冪等性: 各テーブルは INSERT 前に TRUNCATE するため、再実行しても安全。
"""

import argparse

import duckdb

from src.utils.data_path_utils import get_database_url
from src.utils.logger import get_logger

logger = get_logger(__name__)

# テーブル名 → 明示カラムリスト（SELECT * による暗黙のカラム順一致に頼らない）
_TABLES: dict[str, list[str]] = {
    "stock_features": ["market", "symbol", "row_num"],  # 実際は動的列を持つため Step 2 で特別扱い
    "prediction_results": [
        "market", "symbol", "predicted_at", "model_version", "run_id", "current_price",
        "avg_pred_price", "diff_ratio", "model_count", "confidence_ratio",
        "avg_pred_price_3d", "avg_pred_price_5d", "avg_pred_price_10d",
        "diff_ratio_3d", "diff_ratio_5d", "diff_ratio_10d", "confluence_score",
    ],
    "market_data_raw": [
        "market", "symbol", "ticker", "timeframe", "ts", "open", "high", "low",
        "close", "volume", "adj_close", "source", "ingested_at",
    ],
    "index_membership_history": ["market", "symbol", "index_name", "snapshot_date", "source", "fetched_at"],
    "model_metrics": ["market", "symbol", "model_name", "trained_at", "rmse", "directional_accuracy", "n_samples"],
    "prediction_accuracy": [
        "market", "symbol", "model_name", "predicted_at", "horizon", "predicted_price",
        "actual_price", "predicted_ratio", "actual_ratio", "direction_match", "checked_at",
    ],
    "paper_balance": ["balance"],
    "paper_orders": [
        "order_id", "market", "predicted_at", "symbol", "side", "qty", "price", "signal_price",
        "order_type", "status", "fill_price", "realized_pnl", "filled_at", "created_at",
        "horizon", "target_exit_date",
    ],
    "paper_positions": ["symbol", "qty", "avg_price", "updated_at"],
    "shap_values": ["market", "symbol", "model_name", "trained_at", "feature", "shap_mean", "shap_rank"],
    "paper_real_diff": [
        "market", "symbol", "predicted_at", "side", "signal_price", "paper_order_id", "real_order_id",
        "paper_price", "real_price", "paper_slippage", "real_slippage", "price_diff",
        "paper_filled_at", "real_checked_at", "created_at", "updated_at", "order_session", "split_ratio",
    ],
    "feature_selection_log": [
        "market", "symbol", "model_name", "trained_at", "feature", "importance_mean",
        "importance_std", "importance_rank", "is_excluded", "protected_by_shap",
    ],
    "experiment_runs": [
        "run_id", "market", "symbol", "model_name", "trained_at", "horizon", "rmse",
        "directional_accuracy", "n_samples", "n_features", "feature_hash", "params_json", "created_at",
    ],
    "order_run_summary": [
        "run_id", "market", "mode", "run_at", "buy_orders", "sell_orders", "short_orders",
        "skipped", "skipped_min_change", "total_turnover", "min_change_ratio",
    ],
    "paper_short_positions": ["symbol", "qty", "avg_short_price", "unrealized_pnl", "opened_at", "updated_at"],
    "dd_state": ["id", "peak_balance"],
    "data_quality_log": ["market", "symbol", "check_name", "level", "detail", "checked_at"],
    "system_config": ["key", "value", "updated_at"],
    "accuracy_weekly_snapshots": ["week_start", "market", "symbol", "direction_accuracy", "mean_abs_error", "n_samples", "snapshot_at"],
    "earnings_calendar": ["market", "symbol", "event_date", "event_type", "fetched_at"],
    "stock_fundamentals": [
        "market", "symbol", "as_of", "revenue", "operating_income", "net_income", "eps", "roe",
        "op_margin", "net_margin", "debt_to_equity", "cash", "market_cap", "shares_outstanding",
        "revenue_cagr_3y", "fetched_at",
    ],
    "strategy_promotions": ["pr_number", "merge_commit_hash", "rule_or_feature_id", "promoted_at", "pre_promotion_baseline", "status"],
    "factory_runs": [
        "hypothesis_hash", "market", "spec_json", "sharpe_ratio", "win_rate", "num_trades",
        "max_drawdown", "total_return", "dsr", "pbo", "gate_passed", "gate_reasons", "report_path", "evaluated_at",
    ],
    "claude_reasoning": ["run_id", "market", "thinking", "summary", "created_at"],
}
```

- [ ] **Step 2: `stock_features`は動的列を持つため、実際の列を実行時に取得する処理を追加**

`_TABLES`辞書のstock_featuresエントリはプレースホルダに過ぎない。移行関数内で実列を動的取得する:

```python
def _get_dynamic_columns(src_con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    """DuckDB側の実際のカラム一覧を取得する（stock_features等、動的にALTERされるテーブル用）"""
    rows = src_con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()
    return [row[0] for row in rows]
```

- [ ] **Step 3: 移行本体を実装**

```python
def migrate_table(src_con: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> int:
    col_list = ", ".join(f'"{c}"' for c in columns)
    src_con.execute(f'TRUNCATE TABLE pg."{table}"')
    result = src_con.execute(
        f'INSERT INTO pg."{table}" ({col_list}) SELECT {col_list} FROM "{table}"'
    )
    count = src_con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    logger.info(f"移行完了: {table} ({count}行)")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="DuckDB→PostgreSQL 一括データ移行")
    parser.add_argument("--duckdb-path", required=True, help="移行元DuckDBファイルパス")
    args = parser.parse_args()

    src_con = duckdb.connect(args.duckdb_path, read_only=True)
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES)")

    total = 0
    for table, columns in _TABLES.items():
        cols = _get_dynamic_columns(src_con, table) if table == "stock_features" else columns
        total += migrate_table(src_con, table, cols)

    logger.info(f"=== 移行完了: 全 {len(_TABLES)} テーブル、計 {total} 行 ===")
    src_con.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 統合テストを作成**

```python
"""migrate_to_postgres.py の統合テスト（実DuckDB・実Postgres両方を使用）"""

import duckdb
import pytest

from scripts.migrate_to_postgres import migrate_table


@pytest.fixture
def sample_duckdb(tmp_path):
    db_path = str(tmp_path / "sample.duckdb")
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE system_config (key VARCHAR PRIMARY KEY, value VARCHAR, updated_at TIMESTAMP)")
    con.execute("INSERT INTO system_config VALUES ('k1', 'v1', CURRENT_TIMESTAMP)")
    con.close()
    return db_path


def test_migrate_table_copies_all_rows(sample_duckdb, _isolate_db):
    from src.utils.data_path_utils import get_database_url

    src_con = duckdb.connect(sample_duckdb, read_only=True)
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES)")

    count = migrate_table(src_con, "system_config", ["key", "value", "updated_at"])

    assert count == 1
    src_con.close()
```

保存先: `python/tests/integration/scripts/test_migrate_to_postgres.py`。`_isolate_db`（Task 4でintegration/conftest.pyに追加済み）を使い、Postgres側はテスト終了時にロールバックされる。

- [ ] **Step 5: テスト実行**

```bash
cd python && python -m pytest tests/integration/scripts/test_migrate_to_postgres.py -v
```

Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add python/scripts/migrate_to_postgres.py python/tests/integration/scripts/test_migrate_to_postgres.py
git commit -m "feat: DuckDB→PostgreSQLデータ移行スクリプトを追加"
```

---

## Task 15: 整合性検証スクリプト `scripts/verify_postgres_migration.py`

**Files:**
- Create: `python/scripts/verify_postgres_migration.py`
- Test: `python/tests/integration/scripts/test_verify_postgres_migration.py`

- [ ] **Step 1: 検証本体を実装**

各テーブルの`COUNT(*)`をDuckDB側・Postgres側で突き合わせ、`paper_balance`・`paper_positions`・`paper_orders`は金額の合計値も照合する:

```python
"""
DuckDB→PostgreSQL 移行後の整合性検証スクリプト

Usage:
    python scripts/verify_postgres_migration.py --duckdb-path data/stockfixer.duckdb

各テーブルのCOUNT(*)、および金額系テーブルはSUM(realized_pnl)等も突き合わせる。
不一致があれば非ゼロ終了する。
"""

import argparse
import sys

import duckdb

from src.utils.data_path_utils import get_database_url
from src.utils.logger import get_logger
from scripts.migrate_to_postgres import _TABLES, _get_dynamic_columns

logger = get_logger(__name__)

# テーブル名 → 追加で照合する SUM 対象カラム
_SUM_CHECKS: dict[str, list[str]] = {
    "paper_balance": ["balance"],
    "paper_positions": ["qty"],
    "paper_orders": ["realized_pnl"],
}


def verify_table(src_con: duckdb.DuckDBPyConnection, table: str) -> bool:
    src_count = src_con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    pg_count = src_con.execute(f'SELECT COUNT(*) FROM pg."{table}"').fetchone()[0]
    if src_count != pg_count:
        logger.error(f"件数不一致: {table} DuckDB={src_count} Postgres={pg_count}")
        return False

    ok = True
    for col in _SUM_CHECKS.get(table, []):
        src_sum = src_con.execute(f'SELECT COALESCE(SUM("{col}"), 0) FROM "{table}"').fetchone()[0]
        pg_sum = src_con.execute(f'SELECT COALESCE(SUM("{col}"), 0) FROM pg."{table}"').fetchone()[0]
        if abs(float(src_sum) - float(pg_sum)) > 1e-6:
            logger.error(f"合計値不一致: {table}.{col} DuckDB={src_sum} Postgres={pg_sum}")
            ok = False

    logger.info(f"検証OK: {table} ({src_count}行)")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="移行後の整合性検証")
    parser.add_argument("--duckdb-path", required=True)
    args = parser.parse_args()

    src_con = duckdb.connect(args.duckdb_path, read_only=True)
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES)")

    all_ok = True
    for table in _TABLES:
        if not verify_table(src_con, table):
            all_ok = False

    src_con.close()
    if not all_ok:
        logger.error("=== 整合性検証NG。切り替えを中止してください ===")
        return 1
    logger.info("=== 整合性検証OK。切り替え可能です ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: テストを作成**

```python
import duckdb
import pytest

from scripts.verify_postgres_migration import verify_table


@pytest.fixture
def matched_duckdb(tmp_path):
    db_path = str(tmp_path / "sample.duckdb")
    con = duckdb.connect(db_path)
    con.execute("CREATE TABLE dd_state (id INTEGER PRIMARY KEY, peak_balance DOUBLE)")
    con.execute("INSERT INTO dd_state VALUES (1, 500.0)")
    con.close()
    return db_path


def test_verify_table_passes_when_counts_match(matched_duckdb, _isolate_db):
    from src.utils.data_path_utils import get_database_url
    from src.utils.db._connection import _db_connection

    with _db_connection() as con:
        con.execute("INSERT INTO dd_state (id, peak_balance) VALUES (1, 500.0)")

    src_con = duckdb.connect(matched_duckdb, read_only=True)
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES)")

    assert verify_table(src_con, "dd_state") is True
    src_con.close()


def test_verify_table_fails_when_counts_differ(matched_duckdb, _isolate_db):
    from src.utils.data_path_utils import get_database_url

    src_con = duckdb.connect(matched_duckdb, read_only=True)
    src_con.execute(f"ATTACH '{get_database_url()}' AS pg (TYPE POSTGRES)")

    assert verify_table(src_con, "dd_state") is False
    src_con.close()
```

保存先: `python/tests/integration/scripts/test_verify_postgres_migration.py`

- [ ] **Step 3: テスト実行**

```bash
cd python && python -m pytest tests/integration/scripts/test_verify_postgres_migration.py -v
```

Expected: `2 passed`

- [ ] **Step 4: Commit**

```bash
git add python/scripts/verify_postgres_migration.py python/tests/integration/scripts/test_verify_postgres_migration.py
git commit -m "feat: 移行後の整合性検証スクリプトを追加"
```

---

## Task 16: CI配線（GitHub Actions services:postgres）

**Files:**
- Modify: `.github/workflows/`配下、unit/integrationテストを実行するワークフローファイル（実際のファイル名は`git grep -l "pytest tests/unit" .github/workflows/`で特定する）
- Modify: `python/check-ci.ps1`
- Modify: `python/check-ci.sh`

- [ ] **Step 1: 対象ワークフローファイルを特定**

```bash
grep -rl "pytest tests/unit\|pytest tests/integration" .github/workflows/
```

- [ ] **Step 2: `services: postgres` を追加**

特定したworkflowの該当jobに以下を追加する:

```yaml
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: stockfixer
          POSTGRES_USER: stockfixer
          POSTGRES_PASSWORD: stockfixer_ci
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U stockfixer -d stockfixer"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
```

テスト実行ステップの直前（または`env:`ブロック）に以下を追加する:

```yaml
    env:
      DATABASE_URL: postgresql://stockfixer:stockfixer_ci@localhost:5432/stockfixer
```

- [ ] **Step 3: `check-ci.ps1` / `check-ci.sh` にローカルPostgres起動チェックを追加**

`python/check-ci.ps1`のテスト実行部分の直前に以下を追加する:

```powershell
Write-Host "PostgreSQL起動確認..."
docker compose up -d postgres
docker compose exec postgres pg_isready -U stockfixer -d stockfixer
if ($LASTEXITCODE -ne 0) {
    Write-Error "PostgreSQLが起動していません。docker compose up -d postgres を確認してください。"
    exit 1
}
$env:DATABASE_URL = "postgresql://stockfixer:stockfixer_dev@localhost:5432/stockfixer"
```

`python/check-ci.sh`に同等の処理をbash構文で追加する:

```bash
echo "PostgreSQL起動確認..."
docker compose up -d postgres
docker compose exec postgres pg_isready -U stockfixer -d stockfixer || {
    echo "PostgreSQLが起動していません。docker compose up -d postgres を確認してください。" >&2
    exit 1
}
export DATABASE_URL="postgresql://stockfixer:stockfixer_dev@localhost:5432/stockfixer"
```

- [ ] **Step 4: CI相当のローカル確認**

```bash
docker compose up -d postgres
cd python && bash check-ci.sh
```

Expected: 全チェック（lint/mypy/pylint/import-linter/unit tests/bandit/pip-audit）green

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ python/check-ci.ps1 python/check-ci.sh
git commit -m "ci: GitHub ActionsとローカルCIチェックにPostgreSQLサービスを配線"
```

---

## Task 17: 切り替えランブック文書の作成

**Files:**
- Create: `docs/runbooks/postgres_cutover.md`

- [ ] **Step 1: ランブックを作成**

設計書（`docs/superpowers/specs/2026-07-19-postgres-migration-design.md`）のデータ移行手順セクションを、実際に叩くコマンド付きで運用手順書化する:

```markdown
# PostgreSQL 切り替えランブック

対象: DuckDB → PostgreSQL のビッグバング切り替え（本番実施用）

## 前提

- Task 1〜16 が全てdevelopにマージ済みであること
- `docker compose config` でpostgresサービスが定義されていることを確認済み

## 手順

1. メンテナンス開始（スケジューラー・Bot・APIを停止）

   ```bash
   docker compose stop stockfixer
   ```

2. DuckDBバックアップ

   ```bash
   cp python/data/stockfixer.duckdb python/data/backups/stockfixer_pre_postgres_$(date +%Y%m%d).duckdb
   ```

3. Postgres起動 + マイグレーション適用

   ```bash
   docker compose up -d postgres
   docker compose exec postgres pg_isready -U stockfixer -d stockfixer
   # マイグレーションは _connection.py の初回接続時に自動適用されるため、
   # 疎通確認のみで良い（アプリ起動前に手動で確認したい場合は python -c "from src.utils.db import init_tables; init_tables()"）
   ```

4. データ移行スクリプト実行

   ```bash
   cd python
   python scripts/migrate_to_postgres.py --duckdb-path data/stockfixer.duckdb
   ```

5. 整合性検証

   ```bash
   python scripts/verify_postgres_migration.py --duckdb-path data/stockfixer.duckdb
   ```

   終了コード0を確認する。非ゼロの場合は手順を中止し、原因を調査する。

6. 切り替え

   `python/.env`に`DATABASE_URL`を設定し、全プロセスを再起動する。

   ```bash
   docker compose up -d stockfixer
   docker compose logs -f stockfixer
   ```

   起動ログにマイグレーションエラーが無いこと、`/health`エンドポイントが200を返すことを確認する。

7. 保持期間

   `python/data/backups/stockfixer_pre_postgres_*.duckdb`は2週間保持し、問題なければ削除する。

## pg_dumpバックアップの復元手順（Task 12.5で判明した重要な注意点）

Task 12.5で `backup_pipeline.py` を `pg_dump`（カスタムフォーマット, `-Fc`）ベースに移行した際、`stockfixer` アプリコンテナに同梱される `pg_dump`（Debian trixieベースイメージのデフォルト、v17系）と、`postgres` サービスコンテナ（`postgres:16-alpine`、v16系）の間にバージョン差異があることが判明した。

**`pg_dump` のカスタムアーカイブフォーマットは「ダンプしたツールのバージョン」にアーカイブヘッダのバージョンが紐づき、より古い `pg_restore` は新しいアーカイブを読めない**（逆に新しい `pg_restore` は古いアーカイブを読める、という非対称な互換性）。実際に検証したところ、v17の `pg_dump` で作成したダンプを `docker compose exec postgres pg_restore`（v16）で読もうとすると `pg_restore: error: unsupported version (1.16) in file header` で失敗する。

**復元時は必ずダンプを作成したのと同じツール（`stockfixer` アプリコンテナ内の `pg_restore`）を使うこと**:

```bash
# NG: postgresサービスコンテナのpg_restoreはバージョンが古く読めない
docker compose exec postgres pg_restore ...

# OK: stockfixerアプリコンテナのpg_restoreを使う
docker compose exec stockfixer pg_restore -h postgres -U stockfixer -d stockfixer -c /app/data/backups/<timestamp>/stockfixer.dump
```

なお、Task 12.5の検証は `pg_restore --list`（アーカイブのテーブル一覧読み取り）による互換性確認までで、実データを実際に空DBへ復元する完全なリストア手順の実地検証はまだ行っていない。本番復元が必要になった際は、上記コマンドの後に対象テーブルへのデータ反映を目視確認すること。

## ロールバック手順

問題が発生した場合:

```bash
docker compose stop stockfixer
git revert <該当コミット群>  # _connection.py 等をDuckDB版に戻す
# .env の DATABASE_URL を削除
docker compose up -d stockfixer
```

DuckDBファイルはStep 2でバックアップ済みのため、`python/data/stockfixer.duckdb`が
移行スクリプト実行時点のまま残っていることを確認する（移行スクリプトは読み取り専用接続のため元ファイルは変更されない）。
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/postgres_cutover.md
git commit -m "docs: PostgreSQL切り替えランブックを追加"
```

---

## Self-Review Notes（作成者メモ）

- **Spec coverage**: 設計書の①〜⑦全セクションに対応するタスクを配置済み（①②→Task1-3, ③→Task14-15,17, ④→Task5-13, ⑤→Task4, ⑥⑦→Task17・各タスクのロールバック記述）。
- **既知の残課題**: Task 6-11の一部（特にaccuracy.py, prediction_results.pyの動的WHERE構築箇所）は、プレースホルダの個数と`params`リストの対応関係を実装時に慎重に確認する必要がある。これは「実行して既存テストが通るか」で機械的に検証可能なため、各タスクの回帰テストステップに委ねる。
- **型一貫性**: `_bulk.py`の`bulk_insert`/`bulk_upsert`のシグネチャは、Task 5-11の全呼び出し箇所で一貫して使用している。
