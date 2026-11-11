-- 0008_dd_state_per_broker_postgres: DD の基準 dd_state を Broker ごとに分ける (#1787)。
-- 実口座と紙が同じ peak_balance を書き合わないよう、broker_name ごとに1行を持つ。
-- 既存の id = 1 の行は紙 (paper) の行として残す。2回流しても壊れない形にする。
ALTER TABLE dd_state ADD COLUMN IF NOT EXISTS broker_name VARCHAR;
UPDATE dd_state SET broker_name = 'paper' WHERE broker_name IS NULL AND id = 1;
UPDATE dd_state SET broker_name = 'legacy_' || id WHERE broker_name IS NULL;
ALTER TABLE dd_state ALTER COLUMN broker_name SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_dd_state_broker_name ON dd_state (broker_name);
CREATE SEQUENCE IF NOT EXISTS dd_state_id_seq OWNED BY dd_state.id;
SELECT setval('dd_state_id_seq', COALESCE((SELECT MAX(id) FROM dd_state), 0) + 1, false);
ALTER TABLE dd_state ALTER COLUMN id SET DEFAULT nextval('dd_state_id_seq');
