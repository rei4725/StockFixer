-- 0012_add_morning_view_log_postgres: 朝 8:30 の目線の告知と答え合わせの記録 (#1875 段2)。
-- 1日1行（day が主キー、同じ日は上書き）。08:30 に告知が出なかった日は行を作らない。
-- view と futures_view は up か down、source は claude か fallback。
-- actual_change_pct は日経平均の前日の終値比の騰落率で、答え合わせの前は NULL。
-- 時刻は UTC。2回流しても壊れない。
CREATE TABLE IF NOT EXISTS morning_view_log (
    day                DATE      NOT NULL PRIMARY KEY,
    view               VARCHAR   NOT NULL,
    source             VARCHAR   NOT NULL,
    futures_view       VARCHAR   NOT NULL,
    actual_change_pct  DOUBLE PRECISION,
    announced_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    judged_at          TIMESTAMP
);
