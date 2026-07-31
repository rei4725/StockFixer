"""本体側の推論サービスクライアントのテスト。

サービス障害時に例外を投げず None を返す（＝インプロセス推論へフォールバック
する合図）ことを重点的に検証する。
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.prediction.remote_client import get_service_url, predict_via_service


def _call(**overrides):
    kwargs = {
        "market": "jp",
        "symbol": "7203",
        "current_price": 1000.0,
        "features": {"a": 1.0},
        "model_types": ["ModelA"],
        "model_weights": [1.0],
    }
    kwargs.update(overrides)
    return predict_via_service(**kwargs)


class TestGetServiceUrl:
    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("PREDICTION_SERVICE_URL", raising=False)
        assert get_service_url() is None

    def test_returns_url_when_set(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://localhost:5200")
        assert get_service_url() == "http://localhost:5200"

    def test_blank_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "   ")
        assert get_service_url() is None


class TestPredictViaService:
    def test_returns_none_when_url_unset(self, monkeypatch):
        """URL 未設定なら HTTP を呼ばずに None を返す（既定動作）。"""
        monkeypatch.delenv("PREDICTION_SERVICE_URL", raising=False)
        with patch("src.prediction.remote_client.requests.post") as mock_post:
            assert _call() is None
            mock_post.assert_not_called()

    def test_successful_response_mapped_to_prediction_result(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "market": "jp",
            "symbol": "7203",
            "current_price": 1000.0,
            "avg_pred_price": 1020.0,
            "diff_ratio": 0.02,
            "model_count": 2,
            "used_models": ["ModelA", "ModelB"],
        }

        with patch("src.prediction.remote_client.requests.post", return_value=mock_resp):
            result = _call()

        assert result is not None
        assert result.market == "jp"
        assert result.symbol == "7203"
        assert result.avg_pred_price == pytest.approx(1020.0)
        assert result.diff_ratio == pytest.approx(0.02)
        assert result.model_count == 2

    def test_zero_model_count_returns_none(self, monkeypatch):
        """model_count=0 は「予測なし」を意味するので None を返す。"""
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "market": "jp",
            "symbol": "7203",
            "current_price": 1000.0,
            "avg_pred_price": 0.0,
            "diff_ratio": 0.0,
            "model_count": 0,
            "used_models": [],
        }

        with patch("src.prediction.remote_client.requests.post", return_value=mock_resp):
            assert _call() is None

    def test_timeout_returns_none(self, monkeypatch):
        """タイムアウトは例外を伝播させず None を返す（フォールバックの合図）。"""
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        with patch(
            "src.prediction.remote_client.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            assert _call() is None

    def test_connection_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        with patch(
            "src.prediction.remote_client.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            assert _call() is None

    def test_server_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("src.prediction.remote_client.requests.post", return_value=mock_resp):
            assert _call() is None

    def test_malformed_response_returns_none(self, monkeypatch):
        """必須キーが欠けた応答でも例外を投げず None を返すこと。"""
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unexpected": "shape"}

        with patch("src.prediction.remote_client.requests.post", return_value=mock_resp):
            assert _call() is None
