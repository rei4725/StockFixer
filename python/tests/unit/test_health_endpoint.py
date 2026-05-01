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
            patch("src.api.health._check_db", return_value=("ok", None)),
            patch("src.api.health._load_scheduler_last_runs", return_value={}),
            patch("src.api.health._get_last_prediction_at", return_value=None),
        ):
            resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["db"] == "ok"

    def test_health_db_error(self, client):
        """DB接続失敗時に status=degraded・HTTP 503 を返す"""
        with (
            patch("src.api.health._check_db", return_value=("error", "connection refused")),
            patch("src.api.health._load_scheduler_last_runs", return_value={}),
            patch("src.api.health._get_last_prediction_at", return_value=None),
        ):
            resp = client.get("/health")

        assert resp.status_code == 503
        data = resp.get_json()
        assert data["status"] == "degraded"
        assert "error" in data["db"]

    def test_health_includes_scheduler_last_runs(self, client):
        """scheduler_last_runs フィールドがレスポンスに含まれる"""
        runs = {"daily_pipeline": "2026-04-30T07:35:00+00:00"}
        with (
            patch("src.api.health._check_db", return_value=("ok", None)),
            patch("src.api.health._load_scheduler_last_runs", return_value=runs),
            patch("src.api.health._get_last_prediction_at", return_value=None),
        ):
            resp = client.get("/health")

        data = resp.get_json()
        assert data["scheduler_last_runs"] == runs

    def test_health_includes_last_prediction(self, client):
        """last_prediction_at フィールドがレスポンスに含まれる"""
        ts = "2026-04-30T07:40:00"
        with (
            patch("src.api.health._check_db", return_value=("ok", None)),
            patch("src.api.health._load_scheduler_last_runs", return_value={}),
            patch("src.api.health._get_last_prediction_at", return_value=ts),
        ):
            resp = client.get("/health")

        data = resp.get_json()
        assert data["last_prediction_at"] == ts

    def test_health_checked_at_present(self, client):
        """checked_at フィールドが必ず含まれる"""
        with (
            patch("src.api.health._check_db", return_value=("ok", None)),
            patch("src.api.health._load_scheduler_last_runs", return_value={}),
            patch("src.api.health._get_last_prediction_at", return_value=None),
        ):
            resp = client.get("/health")

        data = resp.get_json()
        assert "checked_at" in data
        assert data["checked_at"]  # 空でない


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
