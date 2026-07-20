"""src/api/health.py の単体テスト"""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def client():
    from src.api.health import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        """DB接続成功時に status=ok・HTTP 200 を返す"""
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = (1,)

        with (
            patch("src.api.health._check_db", return_value=("ok", None, None)),
            patch("src.api.health._load_scheduler_last_runs", return_value={}),
        ):
            resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["db"] == "ok"

    def test_health_db_error(self, client):
        """DB接続失敗時に status=degraded・HTTP 503 を返す"""
        with (
            patch(
                "src.api.health._check_db",
                return_value=("error", "connection refused", None),
            ),
            patch("src.api.health._load_scheduler_last_runs", return_value={}),
        ):
            resp = client.get("/health")

        assert resp.status_code == 503
        data = resp.get_json()
        assert data["status"] == "degraded"
        assert "error" in data["db"]

    def test_health_db_busy_returns_ok(self, client):
        """DB が別処理使用中（busy）でも status=ok・HTTP 200 を返す（#550）"""
        with (
            patch("src.api.health._check_db", return_value=("busy", None, None)),
            patch("src.api.health._load_scheduler_last_runs", return_value={}),
        ):
            resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["db"] == "busy"

    def test_health_includes_scheduler_last_runs(self, client):
        """scheduler_last_runs フィールドがレスポンスに含まれる"""
        runs = {"daily_pipeline": "2026-04-30T07:35:00+00:00"}
        with (
            patch("src.api.health._check_db", return_value=("ok", None, None)),
            patch("src.api.health._load_scheduler_last_runs", return_value=runs),
        ):
            resp = client.get("/health")

        data = resp.get_json()
        assert data["scheduler_last_runs"] == runs

    def test_health_includes_last_prediction(self, client):
        """last_prediction_at フィールドがレスポンスに含まれる"""
        ts = "2026-04-30T07:40:00"
        with (
            patch("src.api.health._check_db", return_value=("ok", None, ts)),
            patch("src.api.health._load_scheduler_last_runs", return_value={}),
        ):
            resp = client.get("/health")

        data = resp.get_json()
        assert data["last_prediction_at"] == ts

    def test_health_checked_at_present(self, client):
        """checked_at フィールドが必ず含まれる"""
        with (
            patch("src.api.health._check_db", return_value=("ok", None, None)),
            patch("src.api.health._load_scheduler_last_runs", return_value={}),
        ):
            resp = client.get("/health")

        data = resp.get_json()
        assert "checked_at" in data
        assert data["checked_at"]  # 空でない

    def test_health_degraded_when_scheduler_stale(self, client):
        """スケジューラ最終実行から30分以上経過で status=degraded・HTTP 200 を返す"""
        from datetime import datetime, timedelta, timezone

        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
        runs = {"daily_pipeline": stale_time}
        with (
            patch("src.api.health._check_db", return_value=("ok", None, None)),
            patch("src.api.health._load_scheduler_last_runs", return_value=runs),
        ):
            resp = client.get("/health")

        data = resp.get_json()
        assert data["status"] == "degraded"
        assert resp.status_code == 200  # DB は ok なので HTTP 200

    def test_health_ok_when_scheduler_recent(self, client):
        """スケジューラ最終実行が30分以内なら status=ok を返す"""
        from datetime import datetime, timedelta, timezone

        recent_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        runs = {"daily_pipeline": recent_time}
        with (
            patch("src.api.health._check_db", return_value=("ok", None, None)),
            patch("src.api.health._load_scheduler_last_runs", return_value=runs),
        ):
            resp = client.get("/health")

        data = resp.get_json()
        assert data["status"] == "ok"
        assert resp.status_code == 200


class TestCheckDb:
    """_check_db のロックタイムアウト（busy）と異常（error）の区別（#550/#553）"""

    def test_busy_when_lock_timeout(self):
        """FileLock 取得タイムアウト時は busy を返す（異常扱いしない）"""
        from src.api.health import _check_db
        from src.utils.db._connection import DbLockTimeoutError

        with patch(
            "src.utils.db._connection._db_connection",
            side_effect=DbLockTimeoutError("別プロセスがDBを使用中"),
        ):
            status, err, last_pred = _check_db()

        assert status == "busy"
        assert err is None
        assert last_pred is None

    def test_error_when_connection_fails(self):
        """ロック以外の接続失敗（DB 破損等）は error を返す"""
        from src.api.health import _check_db

        with patch(
            "src.utils.db._connection._db_connection",
            side_effect=RuntimeError("Could not read enough bytes"),
        ):
            status, err, last_pred = _check_db()

        assert status == "error"
        assert "Could not read enough bytes" in err
        assert last_pred is None

    def test_ok_when_db_available(self):
        """DB に接続できれば ok を返す（隔離された一時 DB を使用）"""
        from src.api.health import _check_db

        status, err, last_pred = _check_db()

        assert status == "ok"
        assert err is None
        assert last_pred is None  # 一時 DB に予測データはない

    def test_single_connection_returns_last_prediction(self):
        """1 接続で疎通確認と last_prediction_at をまとめて取得する（#553）"""
        from src.api.health import _check_db
        from src.utils.db._connection import _db_connection

        # 隔離された一時 DB に予測データを 1 件投入
        with _db_connection() as con:
            con.execute(
                "INSERT INTO prediction_results (market, symbol, predicted_at, model_version)"
                " VALUES ('jp', '7203', '20260702_120000', 'production')"
            )

        status, err, last_pred = _check_db()

        assert status == "ok"
        assert err is None
        assert last_pred == "20260702_120000"

    def test_query_last_prediction_swallows_errors(self):
        """last_prediction_at のクエリ失敗は None を返し、例外を伝播させないこと"""
        from src.api.health import _query_last_prediction_at

        bad_con = MagicMock()
        bad_con.execute.side_effect = RuntimeError("query failed")

        assert _query_last_prediction_at(bad_con) is None


class TestDbConnectionLockTimeout:
    """_db_connection の lock_timeout 引数の動作（#550）"""

    def teardown_method(self):
        from src.utils.db._connection import close_connection

        close_connection()

    def test_raises_db_lock_timeout_error_when_pool_exhausted(self):
        """コネクションプール枯渇時に短い lock_timeout で DbLockTimeoutError が出ること"""
        from psycopg_pool import PoolTimeout

        from src.utils.db._connection import DbLockTimeoutError, _db_connection, set_test_connection

        # _isolate_db の注入接続を外し、プール経路を実際に通す
        set_test_connection(None)
        mock_pool = MagicMock()
        mock_pool.connection.side_effect = PoolTimeout("exhausted")

        with patch("src.utils.db._connection._get_pool", return_value=mock_pool):
            with pytest.raises(DbLockTimeoutError):
                with _db_connection(lock_timeout=0.1):
                    pass

    def test_default_timeout_still_raises_runtime_error_compatible(self):
        """DbLockTimeoutError は RuntimeError のサブクラス（既存呼び出し側と互換）"""
        from src.utils.db._connection import DbLockTimeoutError

        assert issubclass(DbLockTimeoutError, RuntimeError)


class TestLoadSchedulerLastRuns:
    def test_returns_empty_when_no_file(self, tmp_path):
        """状態ファイルが存在しない場合は空 dict を返す"""
        with patch("src.api.health.get_results_dir", return_value=str(tmp_path)):
            from src.api.health import _load_scheduler_last_runs

            result = _load_scheduler_last_runs()
        assert result == {}

    def test_returns_latest_success_per_job(self, tmp_path):
        """各 job_id の最新成功時刻のみを返す"""
        state = {
            "events": [
                {
                    "job_id": "daily_pipeline",
                    "status": "success",
                    "finished_at": "2026-04-29T07:35:00+00:00",
                },
                {
                    "job_id": "daily_pipeline",
                    "status": "success",
                    "finished_at": "2026-04-30T07:35:00+00:00",
                },
                {
                    "job_id": "daily_pipeline",
                    "status": "error",
                    "finished_at": "2026-04-30T08:00:00+00:00",
                },
            ]
        }
        state_file = tmp_path / "scheduler_queue_state.json"
        state_file.write_text(json.dumps(state), encoding="utf-8")

        with patch("src.api.health.get_results_dir", return_value=str(tmp_path)):
            from src.api.health import _load_scheduler_last_runs

            result = _load_scheduler_last_runs()

        assert result["daily_pipeline"] == "2026-04-30T07:35:00+00:00"
        assert len(result) == 1  # error イベントは含まれない
