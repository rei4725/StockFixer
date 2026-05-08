"""src/api/metrics.py の単体テスト"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def client():
    from src.api.health import app
    from src.api.metrics import register_routes

    app.config["TESTING"] = True
    register_routes(app)
    with app.test_client() as c:
        yield c


class TestMetricsEndpoint:
    def test_metrics_ok(self, client):
        """/metrics が HTTP 200 と text/plain Content-Type を返す"""
        with patch("src.api.metrics.generate_latest", return_value=b"# HELP\n"):
            resp = client.get("/metrics")

        assert resp.status_code == 200
        assert "text/plain" in resp.content_type

    def test_metrics_body_is_bytes(self, client):
        """レスポンスボディが prometheus テキスト形式（bytes）である"""
        sample = b"# HELP pipeline_duration_seconds\n"
        with patch("src.api.metrics.generate_latest", return_value=sample):
            resp = client.get("/metrics")

        assert sample in resp.data

    def test_register_routes_is_idempotent(self):
        """register_routes を複数回呼んでも例外が発生しない"""
        from src.api.health import app
        from src.api.metrics import register_routes

        register_routes(app)
        register_routes(app)  # 2 回目は no-op


class TestTrackPipeline:
    def test_success_records_duration_and_counter(self):
        """正常終了時に duration と status=success カウンターを記録する"""
        mock_hist = MagicMock()
        mock_counter = MagicMock()

        with (
            patch("src.api.metrics.pipeline_duration", mock_hist),
            patch("src.api.metrics.pipeline_runs_total", mock_counter),
        ):
            from src.api.metrics import track_pipeline

            with track_pipeline("predict"):
                pass

        mock_hist.labels.assert_called_once_with(pipeline="predict")
        mock_hist.labels.return_value.observe.assert_called_once()
        mock_counter.labels.assert_called_once_with(pipeline="predict", status="success")
        mock_counter.labels.return_value.inc.assert_called_once()

    def test_failure_records_fail_status_and_reraises(self):
        """例外時に status=fail を記録し、例外を再 raise する"""
        mock_hist = MagicMock()
        mock_counter = MagicMock()

        with (
            patch("src.api.metrics.pipeline_duration", mock_hist),
            patch("src.api.metrics.pipeline_runs_total", mock_counter),
        ):
            from src.api.metrics import track_pipeline

            with pytest.raises(RuntimeError):
                with track_pipeline("train"):
                    raise RuntimeError("pipeline error")

        mock_hist.labels.assert_called_once_with(pipeline="train")
        mock_hist.labels.return_value.observe.assert_called_once()
        mock_counter.labels.assert_called_once_with(pipeline="train", status="fail")
        mock_counter.labels.return_value.inc.assert_called_once()

    def test_elapsed_time_is_positive(self):
        """記録される実行時間が正の値である"""
        observed_values: list[float] = []

        mock_hist = MagicMock()
        mock_hist.labels.return_value.observe.side_effect = lambda v: observed_values.append(v)
        mock_counter = MagicMock()

        with (
            patch("src.api.metrics.pipeline_duration", mock_hist),
            patch("src.api.metrics.pipeline_runs_total", mock_counter),
        ):
            from src.api.metrics import track_pipeline

            with track_pipeline("backtest"):
                pass

        assert len(observed_values) == 1
        assert observed_values[0] >= 0.0
