-- 0001_initial: 初期スキーマ全テーブル定義
-- _connection.py の _init_tables() に相当する DDL を正式化

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
    current_price       DOUBLE,
    avg_pred_price      DOUBLE,
    diff_ratio          DOUBLE,
    model_count         INTEGER,
    confidence_ratio    DOUBLE,
    avg_pred_price_3d   DOUBLE,
    avg_pred_price_5d   DOUBLE,
    avg_pred_price_10d  DOUBLE,
    diff_ratio_3d       DOUBLE,
    diff_ratio_5d       DOUBLE,
    diff_ratio_10d      DOUBLE,
    confluence_score    INTEGER,
    PRIMARY KEY (market, symbol, predicted_at, model_version)
);

ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS avg_pred_price_3d DOUBLE;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS avg_pred_price_5d DOUBLE;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS avg_pred_price_10d DOUBLE;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS diff_ratio_3d DOUBLE;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS diff_ratio_5d DOUBLE;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS diff_ratio_10d DOUBLE;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS confluence_score INTEGER;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS confidence_ratio DOUBLE;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS model_version VARCHAR;
ALTER TABLE prediction_results ADD COLUMN IF NOT EXISTS run_id VARCHAR;

CREATE TABLE IF NOT EXISTS market_data_raw (
    market      VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    ticker      VARCHAR NOT NULL,
    timeframe   VARCHAR NOT NULL,
    ts          TIMESTAMP NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      BIGINT,
    adj_close   DOUBLE,
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
    rmse                 DOUBLE,
    directional_accuracy DOUBLE,
    n_samples            INTEGER,
    PRIMARY KEY (market, symbol, model_name, trained_at)
);

CREATE TABLE IF NOT EXISTS prediction_accuracy (
    market          VARCHAR NOT NULL,
    symbol          VARCHAR NOT NULL,
    model_name      VARCHAR NOT NULL,
    predicted_at    VARCHAR NOT NULL,
    horizon         INTEGER NOT NULL DEFAULT 1,
    predicted_price DOUBLE,
    actual_price    DOUBLE,
    predicted_ratio DOUBLE,
    actual_ratio    DOUBLE,
    direction_match BOOLEAN,
    checked_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, symbol, model_name, predicted_at, horizon)
);

CREATE TABLE IF NOT EXISTS paper_balance (
    balance DOUBLE NOT NULL
);

INSERT INTO paper_balance
SELECT 1000000.0
WHERE (SELECT COUNT(*) FROM paper_balance) = 0;

CREATE TABLE IF NOT EXISTS paper_orders (
    order_id     VARCHAR NOT NULL PRIMARY KEY,
    market       VARCHAR,
    predicted_at VARCHAR,
    symbol       VARCHAR NOT NULL,
    side         INTEGER NOT NULL,
    qty          INTEGER NOT NULL,
    price        DOUBLE,
    signal_price DOUBLE,
    order_type   INTEGER NOT NULL,
    status       VARCHAR NOT NULL DEFAULT 'pending',
    fill_price   DOUBLE,
    realized_pnl DOUBLE,
    filled_at    TIMESTAMP,
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS realized_pnl DOUBLE;
ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS market VARCHAR;
ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS predicted_at VARCHAR;
ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS signal_price DOUBLE;

CREATE TABLE IF NOT EXISTS paper_positions (
    symbol      VARCHAR NOT NULL PRIMARY KEY,
    qty         INTEGER NOT NULL,
    avg_price   DOUBLE NOT NULL,
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shap_values (
    market      VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    model_name  VARCHAR NOT NULL,
    trained_at  VARCHAR NOT NULL,
    feature     VARCHAR NOT NULL,
    shap_mean   DOUBLE NOT NULL,
    shap_rank   INTEGER NOT NULL,
    PRIMARY KEY (market, symbol, model_name, trained_at, feature)
);

CREATE TABLE IF NOT EXISTS paper_real_diff (
    market          VARCHAR NOT NULL,
    symbol          VARCHAR NOT NULL,
    predicted_at    VARCHAR NOT NULL,
    side            INTEGER NOT NULL,
    signal_price    DOUBLE,
    paper_order_id  VARCHAR,
    real_order_id   VARCHAR,
    paper_price     DOUBLE,
    real_price      DOUBLE,
    paper_slippage  DOUBLE,
    real_slippage   DOUBLE,
    price_diff      DOUBLE,
    paper_filled_at TIMESTAMP,
    real_checked_at TIMESTAMP,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    order_session   VARCHAR,
    split_ratio     DOUBLE,
    PRIMARY KEY (market, symbol, predicted_at, side)
);

ALTER TABLE paper_real_diff ADD COLUMN IF NOT EXISTS order_session VARCHAR;
ALTER TABLE paper_real_diff ADD COLUMN IF NOT EXISTS split_ratio DOUBLE;

CREATE TABLE IF NOT EXISTS feature_selection_log (
    market             VARCHAR NOT NULL,
    symbol             VARCHAR NOT NULL,
    model_name         VARCHAR NOT NULL,
    trained_at         VARCHAR NOT NULL,
    feature            VARCHAR NOT NULL,
    importance_mean    DOUBLE NOT NULL,
    importance_std     DOUBLE NOT NULL,
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
    rmse                 DOUBLE,
    directional_accuracy DOUBLE,
    n_samples            INTEGER,
    n_features           INTEGER,
    feature_hash         VARCHAR,
    params_json          VARCHAR,
    created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_run_summary (
    run_id            VARCHAR   NOT NULL PRIMARY KEY,
    market            VARCHAR   NOT NULL,
    mode              VARCHAR   NOT NULL,
    run_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    buy_orders        INTEGER   NOT NULL DEFAULT 0,
    sell_orders       INTEGER   NOT NULL DEFAULT 0,
    short_orders      INTEGER   NOT NULL DEFAULT 0,
    skipped           INTEGER   NOT NULL DEFAULT 0,
    skipped_min_change INTEGER  NOT NULL DEFAULT 0,
    total_turnover    DOUBLE    NOT NULL DEFAULT 0.0,
    min_change_ratio  DOUBLE
);

CREATE TABLE IF NOT EXISTS paper_short_positions (
    symbol            VARCHAR   NOT NULL PRIMARY KEY,
    qty               INTEGER   NOT NULL,
    avg_short_price   DOUBLE    NOT NULL,
    unrealized_pnl    DOUBLE,
    opened_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dd_state (
    id            INTEGER PRIMARY KEY,
    peak_balance  DOUBLE  NOT NULL
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
)
