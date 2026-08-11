-- 0004_add_fundamentals_value_fields: stock_fundamentals にPER・配当性向を追加
-- バリュー・スクリーナー（低PER・低配当性向・財務安定）向けのフィールド。
-- 既存行に対しては NULL で埋まる（次回 run_fetch_fundamentals.py 実行で充填される）。
ALTER TABLE stock_fundamentals ADD COLUMN IF NOT EXISTS trailing_pe DOUBLE PRECISION;
ALTER TABLE stock_fundamentals ADD COLUMN IF NOT EXISTS payout_ratio DOUBLE PRECISION;
