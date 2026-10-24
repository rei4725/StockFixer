-- 0007_add_symbol_master_postgres: 銘柄コードと社名の対応表 (#1071)。
-- 日本株は JPX の東証上場銘柄一覧から run_symbol_master.py で手動更新する。
-- source に出どころ、updated_at に取得日時 (UTC) を残す。
CREATE TABLE IF NOT EXISTS symbol_master (
    market      VARCHAR   NOT NULL,
    symbol      VARCHAR   NOT NULL,
    name_ja     VARCHAR,
    name_en     VARCHAR,
    sector      VARCHAR,
    source      VARCHAR   NOT NULL,
    updated_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, symbol)
);
