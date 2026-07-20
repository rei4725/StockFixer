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
