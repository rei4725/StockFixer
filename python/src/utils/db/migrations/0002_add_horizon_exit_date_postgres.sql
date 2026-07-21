-- 0002_add_horizon_exit_date_postgres: paper_orders にホライズン情報と強制決済日を追加
-- (0001_baseline_postgres.sql に既に含まれているため、通常は no-op)
ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS horizon INTEGER;
ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS target_exit_date DATE;
