-- 0002_add_horizon_exit_date: paper_orders にホライズン情報と強制決済日を追加
ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS horizon INTEGER;
ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS target_exit_date DATE
