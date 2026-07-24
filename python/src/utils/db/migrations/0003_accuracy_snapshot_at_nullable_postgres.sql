-- 0003_accuracy_snapshot_at_nullable_postgres: accuracy_weekly_snapshots.snapshot_at を nullable に変更
-- DuckDB側では元々nullable列だったが、Postgresベースラインスキーマでは
-- NOT NULL DEFAULT CURRENT_TIMESTAMP としていた。実データではアプリ側が
-- snapshot_at を設定しないまま書き込む期間があり（2026-06-15週以降の大半）、
-- 本番切り替え時のデータ移行でNOT NULL制約違反が発生したため、DuckDB側の
-- 実態に合わせてnullableへ緩和する（データを一件も失わないため）。
ALTER TABLE accuracy_weekly_snapshots ALTER COLUMN snapshot_at DROP NOT NULL;
