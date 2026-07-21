"""診断ログ retention の単体テスト（DB スリム化）。

- 保持日数より古い行は削除される
- ただし各 (market, symbol, model_name) の最新 trained_at は常に残る
  （読み出しが MAX(trained_at) しか見ないため壊さない）

Postgres移行前は本テストは実DBと無関係な in-memory DuckDB 接続を直接
生成して検証していたが、``retention.py`` が ``%s`` プレースホルダ
（psycopg3/Postgres専用構文）に移行されたため、DuckDB接続に対しては
構文エラーになり成立しなくなった。テスト全体を1トランザクションに包み
テスト終了時にロールバックする ``_isolate_db`` フィクスチャ（conftest.py）
経由で実Postgres接続を使うよう書き換え、あわせて最終アサーションにも
market/symbol/model_name の絞り込みを追加した（対象テーブルが常に空である
ことを前提にしない）。
"""

from datetime import datetime, timezone

import pytest

from src.utils.db._connection import _db_connection
from src.utils.db.retention import purge_old_training_logs


def _get_con():
    """テスト用の共有Postgres接続を取得する（_isolate_db フィクスチャが注入したもの）。"""
    with _db_connection() as con:
        return con


def _ins_shap(con, market, symbol, model, trained_at):
    con.execute(
        """
        INSERT INTO shap_values
            (market, symbol, model_name, trained_at, feature, shap_mean, shap_rank)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        [market, symbol, model, trained_at, "f1", 0.1, 1],
    )


def _ins_metrics(con, market, symbol, model, trained_at):
    con.execute(
        """
        INSERT INTO model_metrics
            (market, symbol, model_name, trained_at, rmse, directional_accuracy, n_samples)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        [market, symbol, model, trained_at, 0.01, 0.55, 100],
    )


def _ins_quality(con, market, symbol, check_name, checked_at):
    con.execute(
        """
        INSERT INTO data_quality_log
            (market, symbol, check_name, level, detail, checked_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        [market, symbol, check_name, "WARN", "detail", checked_at],
    )


class TestPurgeOldTrainingLogs:
    def test_deletes_old_but_keeps_latest_per_group(self):
        con = _get_con()
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
            """
            SELECT symbol, trained_at FROM shap_values
            WHERE market = %s AND model_name = %s AND symbol IN (%s, %s)
            ORDER BY symbol, trained_at
            """,
            ["us", "XGB", "AAPL", "MSFT"],
        ).fetchall()
        # AAPL: 古い2件削除 → 最新のみ残る。MSFT: 全部古いが最新1件は残る
        assert rows == [
            ("AAPL", "20260601_000000"),
            ("MSFT", "20260103_000000"),
        ]
        assert deleted["shap_values"] == 3  # AAPL古2 + MSFT古1

    def test_recent_rows_are_kept(self):
        con = _get_con()
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        # 保持窓内（直近）の行は最新でなくても残る
        _ins_shap(con, "us", "AAPL", "XGB", "20260605_000000")
        _ins_shap(con, "us", "AAPL", "XGB", "20260606_000000")
        deleted = purge_old_training_logs(con, retention_days=30, now=now)
        assert deleted["shap_values"] == 0
        n = con.execute(
            "SELECT COUNT(*) FROM shap_values "
            "WHERE market = %s AND symbol = %s AND model_name = %s",
            ["us", "AAPL", "XGB"],
        ).fetchone()[0]
        assert n == 2

    def test_separate_models_kept_independently(self):
        con = _get_con()
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        _ins_shap(con, "us", "AAPL", "XGB", "20260101_000000")
        _ins_shap(con, "us", "AAPL", "LGB", "20260102_000000")
        purge_old_training_logs(con, retention_days=30, now=now)
        # 各 model_name グループの最新が残る → 2件とも残存
        n = con.execute(
            "SELECT COUNT(*) FROM shap_values "
            "WHERE market = %s AND symbol = %s AND model_name IN (%s, %s)",
            ["us", "AAPL", "XGB", "LGB"],
        ).fetchone()[0]
        assert n == 2

    def test_negative_retention_raises(self):
        con = _get_con()
        with pytest.raises(ValueError):
            purge_old_training_logs(con, retention_days=-1)


class TestModelMetricsRetention:
    """model_metrics: 読み出し（load_model_weights）は各 (market, symbol, model_name)
    の最新 trained_at だけ見るため、最新を残して古い行を消すのは安全。
    """

    def test_deletes_old_but_keeps_latest_per_group(self):
        con = _get_con()
        # AAPL/XGB: 古2 + 新1
        _ins_metrics(con, "us", "AAPL", "XGB", "20260101_000000")
        _ins_metrics(con, "us", "AAPL", "XGB", "20260102_000000")
        _ins_metrics(con, "us", "AAPL", "XGB", "20260601_000000")  # 最新
        # MSFT/XGB: 全て古い → 最新1件は残す
        _ins_metrics(con, "us", "MSFT", "XGB", "20260101_000000")
        _ins_metrics(con, "us", "MSFT", "XGB", "20260103_000000")  # この銘柄の最新

        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        deleted = purge_old_training_logs(con, retention_days=30, now=now)

        rows = con.execute(
            """
            SELECT symbol, trained_at FROM model_metrics
            WHERE market = %s AND model_name = %s AND symbol IN (%s, %s)
            ORDER BY symbol, trained_at
            """,
            ["us", "XGB", "AAPL", "MSFT"],
        ).fetchall()
        assert rows == [
            ("AAPL", "20260601_000000"),
            ("MSFT", "20260103_000000"),
        ]
        assert deleted["model_metrics"] == 3  # AAPL古2 + MSFT古1

    def test_latest_per_group_is_max_trained_at(self):
        """読み手 load_model_weights が参照する MAX(trained_at) が retention 後も保たれる。"""
        con = _get_con()
        _ins_metrics(con, "us", "AAPL", "XGB", "20260101_000000")
        _ins_metrics(con, "us", "AAPL", "XGB", "20260301_000000")  # 最新
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        purge_old_training_logs(con, retention_days=30, now=now)
        latest = con.execute(
            "SELECT trained_at FROM model_metrics "
            "WHERE market=%s AND symbol=%s AND model_name=%s "
            "ORDER BY trained_at DESC LIMIT 1",
            ["us", "AAPL", "XGB"],
        ).fetchone()[0]
        assert latest == "20260301_000000"


class TestDataQualityLogRetention:
    """data_quality_log: checked_at は ISO 8601 文字列。各 (market, symbol, check_name)
    の最新を残して古い行を削除する。
    """

    def test_deletes_old_but_keeps_latest_per_group(self):
        con = _get_con()
        # AAPL/gap: 古2 + 新1
        _ins_quality(con, "us", "AAPL", "gap", "2026-01-01T00:00:00+00:00")
        _ins_quality(con, "us", "AAPL", "gap", "2026-01-02T00:00:00+00:00")
        _ins_quality(con, "us", "AAPL", "gap", "2026-06-01T00:00:00+00:00")  # 最新
        # MSFT/nan: 全て古い → 最新1件は残す
        _ins_quality(con, "us", "MSFT", "nan", "2026-01-01T00:00:00+00:00")
        _ins_quality(con, "us", "MSFT", "nan", "2026-01-03T00:00:00+00:00")  # 最新

        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        deleted = purge_old_training_logs(con, retention_days=30, now=now)

        rows = con.execute(
            """
            SELECT symbol, checked_at FROM data_quality_log
            WHERE market = %s AND symbol IN (%s, %s)
              AND check_name IN (%s, %s)
            ORDER BY symbol, checked_at
            """,
            ["us", "AAPL", "MSFT", "gap", "nan"],
        ).fetchall()
        assert rows == [
            ("AAPL", "2026-06-01T00:00:00+00:00"),
            ("MSFT", "2026-01-03T00:00:00+00:00"),
        ]
        assert deleted["data_quality_log"] == 3  # AAPL古2 + MSFT古1

    def test_separate_check_names_kept_independently(self):
        con = _get_con()
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        _ins_quality(con, "us", "AAPL", "gap", "2026-01-01T00:00:00+00:00")
        _ins_quality(con, "us", "AAPL", "nan", "2026-01-02T00:00:00+00:00")
        purge_old_training_logs(con, retention_days=30, now=now)
        # check_name ごとの最新が残る → 2件とも残存
        n = con.execute(
            "SELECT COUNT(*) FROM data_quality_log "
            "WHERE market = %s AND symbol = %s AND check_name IN (%s, %s)",
            ["us", "AAPL", "gap", "nan"],
        ).fetchone()[0]
        assert n == 2

    def test_recent_iso_rows_are_kept(self):
        con = _get_con()
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)
        _ins_quality(con, "us", "AAPL", "gap", "2026-06-05T00:00:00+00:00")
        _ins_quality(con, "us", "AAPL", "gap", "2026-06-06T00:00:00+00:00")
        deleted = purge_old_training_logs(con, retention_days=30, now=now)
        assert deleted["data_quality_log"] == 0
        n = con.execute(
            "SELECT COUNT(*) FROM data_quality_log "
            "WHERE market = %s AND symbol = %s AND check_name = %s",
            ["us", "AAPL", "gap"],
        ).fetchone()[0]
        assert n == 2
