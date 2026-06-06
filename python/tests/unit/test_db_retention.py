"""診断ログ retention の単体テスト（DB スリム化）。

- 保持日数より古い行は削除される
- ただし各 (market, symbol, model_name) の最新 trained_at は常に残る
  （読み出しが MAX(trained_at) しか見ないため壊さない）
"""

from datetime import datetime, timezone

import duckdb
import pytest

from src.utils.db.retention import purge_old_training_logs


def _make_con():
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE shap_values (
            market VARCHAR, symbol VARCHAR, model_name VARCHAR,
            trained_at VARCHAR, feature VARCHAR, shap_mean DOUBLE, shap_rank INTEGER
        )
        """)
    con.execute("""
        CREATE TABLE feature_selection_log (
            market VARCHAR, symbol VARCHAR, model_name VARCHAR,
            trained_at VARCHAR, feature VARCHAR, importance_mean DOUBLE,
            importance_std DOUBLE, importance_rank INTEGER,
            is_excluded BOOLEAN, protected_by_shap BOOLEAN
        )
        """)
    return con


def _ins_shap(con, market, symbol, model, trained_at):
    con.execute(
        "INSERT INTO shap_values VALUES (?,?,?,?,?,?,?)",
        [market, symbol, model, trained_at, "f1", 0.1, 1],
    )


class TestPurgeOldTrainingLogs:
    def test_deletes_old_but_keeps_latest_per_group(self):
        con = _make_con()
        # us/AAPL/XGB: 古い2件 + 新しい1件
        _ins_shap(con, "us", "AAPL", "XGB", "20260101_000000")  # 古
        _ins_shap(con, "us", "AAPL", "XGB", "20260102_000000")  # 古
        _ins_shap(con, "us", "AAPL", "XGB", "20260601_000000")  # 新（最新）
        # us/MSFT/XGB: 全て古い（=最新も古い）→ 最新は残すべき
        _ins_shap(con, "us", "MSFT", "XGB", "20260101_000000")
        _ins_shap(con, "us", "MSFT", "XGB", "20260103_000000")  # この銘柄の最新

        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        deleted = purge_old_training_logs(con, retention_days=30, now=now)

        rows = con.execute(
            "SELECT symbol, trained_at FROM shap_values ORDER BY symbol, trained_at"
        ).fetchall()
        # AAPL: 古い2件削除 → 最新のみ残る。MSFT: 全部古いが最新1件は残る
        assert rows == [
            ("AAPL", "20260601_000000"),
            ("MSFT", "20260103_000000"),
        ]
        assert deleted["shap_values"] == 3  # AAPL古2 + MSFT古1

    def test_recent_rows_are_kept(self):
        con = _make_con()
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        # 保持窓内（直近）の行は最新でなくても残る
        _ins_shap(con, "us", "AAPL", "XGB", "20260605_000000")
        _ins_shap(con, "us", "AAPL", "XGB", "20260606_000000")
        deleted = purge_old_training_logs(con, retention_days=30, now=now)
        assert deleted["shap_values"] == 0
        assert con.execute("SELECT COUNT(*) FROM shap_values").fetchone()[0] == 2

    def test_separate_models_kept_independently(self):
        con = _make_con()
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        _ins_shap(con, "us", "AAPL", "XGB", "20260101_000000")
        _ins_shap(con, "us", "AAPL", "LGB", "20260102_000000")
        purge_old_training_logs(con, retention_days=30, now=now)
        # 各 model_name グループの最新が残る → 2件とも残存
        assert con.execute("SELECT COUNT(*) FROM shap_values").fetchone()[0] == 2

    def test_negative_retention_raises(self):
        con = _make_con()
        with pytest.raises(ValueError):
            purge_old_training_logs(con, retention_days=-1)
