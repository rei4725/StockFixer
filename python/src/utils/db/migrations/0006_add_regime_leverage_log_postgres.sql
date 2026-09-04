-- 0006_add_regime_leverage_log_postgres: STRATEGY.md 7章(強気相場・レバレッジ買い持ち)
-- ペーパートレードの状態を追記専用ログとして記録するテーブル。id最大の行が現在の
-- 建玉・評価額の状態を表す。allocation_rebalance_logと異なり、週次(レジーム判定)と
-- 日次(マージンコール判定)の2種類のジョブが同じ状態を読み書きするため、action/reason
-- で発生源を区別する。
CREATE TABLE IF NOT EXISTS regime_leverage_log (
    id                    BIGSERIAL PRIMARY KEY,
    executed_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    action                VARCHAR NOT NULL,
    reason                VARCHAR NOT NULL,
    spy_price_usd         DOUBLE PRECISION NOT NULL,
    usdjpy_rate           DOUBLE PRECISION NOT NULL,
    shares                DOUBLE PRECISION NOT NULL,
    entry_date            TIMESTAMP,
    entry_price_jpy       DOUBLE PRECISION,
    entry_commission_jpy  DOUBLE PRECISION,
    equity_at_entry_jpy   DOUBLE PRECISION,
    stop_price_jpy        DOUBLE PRECISION,
    equity_now_jpy        DOUBLE PRECISION NOT NULL,
    maintenance_ratio     DOUBLE PRECISION
);
