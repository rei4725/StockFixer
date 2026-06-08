"""DB コンパクション（再構築コピー）の単体テスト。

- 診断ログは retention 条件で絞られる（直近＋各グループ最新を保持）
- それ以外のテーブルは全行コピー
- 列の型・名前が保持される
"""

import os
from datetime import datetime, timezone

import duckdb

from src.utils.db.compact import compact_database, compact_in_place, swap_compacted


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


class TestSwapCompacted:
    def test_swap_keeps_backup(self, tmp_path):
        db = tmp_path / "db.duckdb"
        new = tmp_path / "db.duckdb.compact"
        db.write_text("OLD")
        new.write_text("NEW")

        now = datetime(2026, 6, 1, 3, 0, 0, tzinfo=timezone.utc)
        bak = swap_compacted(str(db), str(new), keep_backup=True, now=now)

        assert db.read_text() == "NEW"  # 新ファイルが本体に
        assert not new.exists()  # .compact は消費された
        assert bak is not None and os.path.exists(bak)
        assert open(bak).read() == "OLD"  # 退避に旧ファイル

    def test_swap_removes_backup_when_not_kept(self, tmp_path):
        db = tmp_path / "db.duckdb"
        new = tmp_path / "db.duckdb.compact"
        db.write_text("OLD")
        new.write_text("NEW")

        bak = swap_compacted(str(db), str(new), keep_backup=False)

        assert db.read_text() == "NEW"
        assert bak is None
        # 退避ファイルは残らない
        assert not list(tmp_path.glob("*.bak-*"))


class TestCompactInPlace:
    def test_rebuilds_and_swaps_in_place(self, tmp_path):
        db = str(tmp_path / "db.duckdb")
        _build_src(db)
        size_before = os.path.getsize(db)

        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        counts = compact_in_place(db, retention_days=30, keep_backup=False, now=now)

        # 同じパスにコンパクション結果が入っている
        assert counts["shap_values"] == (5, 2)
        assert counts["prediction_results"] == (10, 10)
        # 一時ファイル・退避ファイルは残らない
        assert not os.path.exists(db + ".compact")
        assert not list(tmp_path.glob("*.bak-*"))
        # 入れ替え後の DB が読め、フィルタ結果が反映されている
        con = duckdb.connect(db, read_only=True)
        try:
            assert con.execute("SELECT COUNT(*) FROM shap_values").fetchone()[0] == 2
            assert con.execute("SELECT COUNT(*) FROM prediction_results").fetchone()[0] == 10
        finally:
            con.close()
        # ファイルは肥大していない（再構築後 ≤ 元）
        assert os.path.getsize(db) <= size_before

    def test_keep_backup_preserves_original(self, tmp_path):
        db = str(tmp_path / "db.duckdb")
        _build_src(db)
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        compact_in_place(db, retention_days=30, keep_backup=True, now=now)
        baks = list(tmp_path.glob("*.bak-*"))
        assert len(baks) == 1  # 退避ファイルが残る
