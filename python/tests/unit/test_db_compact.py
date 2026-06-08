"""DB コンパクション（再構築コピー）の単体テスト。

- 診断ログは retention 条件で絞られる（直近＋各グループ最新を保持）
- それ以外のテーブルは全行コピー
- 列の型・名前が保持される
"""

import os
from datetime import datetime, timezone

import duckdb

from src.utils.db.compact import compact_database


def _build_src(path):
    con = duckdb.connect(path)
    con.execute("""CREATE TABLE shap_values (
            market VARCHAR, symbol VARCHAR, model_name VARCHAR,
            trained_at VARCHAR, feature VARCHAR, shap_mean DOUBLE, shap_rank INTEGER)""")
    # AAPL: 古2 + 新1、MSFT: 古2（最新も古い）
    rows = [
        ("us", "AAPL", "XGB", "20260101_000000"),
        ("us", "AAPL", "XGB", "20260102_000000"),
        ("us", "AAPL", "XGB", "20260601_000000"),
        ("us", "MSFT", "XGB", "20260101_000000"),
        ("us", "MSFT", "XGB", "20260103_000000"),
    ]
    for m, s, md, ta in rows:
        con.execute("INSERT INTO shap_values VALUES (?,?,?,?,?,?,?)", [m, s, md, ta, "f1", 0.1, 1])
    # 通常テーブル（全コピーされるべき）
    con.execute(
        "CREATE TABLE prediction_results (run_timestamp VARCHAR, market VARCHAR, val DOUBLE)"
    )
    for i in range(10):
        con.execute("INSERT INTO prediction_results VALUES (?,?,?)", [f"t{i}", "us", float(i)])
    con.execute("CHECKPOINT")
    con.close()


class TestCompactDatabase:
    def test_filters_logs_keeps_others_and_schema(self, tmp_path):
        src = str(tmp_path / "src.duckdb")
        dst = str(tmp_path / "dst.duckdb")
        _build_src(src)

        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        counts = compact_database(src, dst, retention_days=30, now=now)

        # shap: AAPL 古2削除→最新1、MSFT 全古だが最新1保持 = 2件
        assert counts["shap_values"] == (5, 2)
        # 通常テーブルは全コピー
        assert counts["prediction_results"] == (10, 10)

        con = duckdb.connect(dst, read_only=True)
        try:
            kept = con.execute(
                "SELECT symbol, trained_at FROM shap_values ORDER BY symbol"
            ).fetchall()
            assert kept == [("AAPL", "20260601_000000"), ("MSFT", "20260103_000000")]
            # 列名・型が保持されている
            desc = {r[0]: r[1] for r in con.execute("DESCRIBE shap_values").fetchall()}
            assert desc["shap_mean"] == "DOUBLE"
            assert desc["shap_rank"] == "INTEGER"
            assert con.execute("SELECT COUNT(*) FROM prediction_results").fetchone()[0] == 10
        finally:
            con.close()

    def test_output_file_created(self, tmp_path):
        src = str(tmp_path / "src.duckdb")
        dst = str(tmp_path / "dst.duckdb")
        _build_src(src)
        compact_database(src, dst, retention_days=30, now=datetime(2026, 6, 6, tzinfo=timezone.utc))
        assert os.path.exists(dst)
