-- 0005_add_allocation_rebalance_log_postgres: 配分戦略(TQQQ/短期債)ペーパートレードの
-- 状態を追記専用ログとして記録するテーブル。id最大の行が現在の建玉・現金の状態を表す。
CREATE TABLE IF NOT EXISTS allocation_rebalance_log (
    id              BIGSERIAL PRIMARY KEY,
    executed_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    action          VARCHAR NOT NULL,
    tqqq_price      DOUBLE PRECISION NOT NULL,
    shy_price       DOUBLE PRECISION NOT NULL,
    tqqq_qty_before DOUBLE PRECISION NOT NULL,
    shy_qty_before  DOUBLE PRECISION NOT NULL,
    cash_before     DOUBLE PRECISION NOT NULL,
    tqqq_qty_after  DOUBLE PRECISION NOT NULL,
    shy_qty_after   DOUBLE PRECISION NOT NULL,
    cash_after      DOUBLE PRECISION NOT NULL
);
