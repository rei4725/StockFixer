-- 0009_rule_best_sell_only_postgres: rule_best_by_symbol に売りだけの印を足す (#1797)。
-- 週次の評価で有効ルールが無いかウォッチリストの外でも保有中だからと残した行に立て、
-- その行から新規買いを出さない。今ある行は印なし（FALSE）のまま残す。2回流しても壊れない。
ALTER TABLE rule_best_by_symbol ADD COLUMN IF NOT EXISTS sell_only BOOLEAN NOT NULL DEFAULT FALSE;
