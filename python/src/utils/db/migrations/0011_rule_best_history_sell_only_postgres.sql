-- 0011_rule_best_history_sell_only_postgres: rule_best_history に売りだけの印を足す (#1889)。
-- rule_best_by_symbol の sell_only (0009) を週の履歴へ写し、週のルールの出入りと
-- backtest_gap が売りだけの行を選ばれたルールとして数えないようにする。
-- 表は scoring の ensure_scoring_tables が作るので、まだ無ければ何もしない。
-- 今ある行は印なし（FALSE）のまま残す。2回流しても壊れない。
ALTER TABLE IF EXISTS rule_best_history ADD COLUMN IF NOT EXISTS sell_only BOOLEAN NOT NULL DEFAULT FALSE;
