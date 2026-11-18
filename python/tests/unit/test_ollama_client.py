"""OllamaClient.is_available の戻り値と警告ログのテスト（#2008）。"""

from unittest.mock import MagicMock, patch

import requests

from src.utils.ollama_client import OllamaClient

_URL = "http://ollama:11434"


def _resp(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    return resp


class TestIsAvailable:
    def test_no_url_returns_false_without_request(self) -> None:
        client = OllamaClient(base_url="")
        with patch("src.utils.ollama_client.requests.get") as mock_get, patch(
            "src.utils.ollama_client.logger"
        ) as mock_logger:
            assert client.is_available is False
        mock_get.assert_not_called()
        mock_logger.warning.assert_not_called()

    def test_ok_returns_true_without_warning(self) -> None:
        client = OllamaClient(base_url=_URL)
        with patch(
            "src.utils.ollama_client.requests.get", return_value=_resp(200)
        ) as mock_get, patch("src.utils.ollama_client.logger") as mock_logger:
            assert client.is_available is True
        mock_get.assert_called_once_with(f"{_URL}/api/tags", timeout=5)
        mock_logger.warning.assert_not_called()

    def test_non_200_returns_false(self) -> None:
        client = OllamaClient(base_url=_URL)
        with patch("src.utils.ollama_client.requests.get", return_value=_resp(503)):
            assert client.is_available is False

    def test_non_200_logs_warning_with_status(self) -> None:
        client = OllamaClient(base_url=_URL)
        with patch("src.utils.ollama_client.requests.get", return_value=_resp(503)), patch(
            "src.utils.ollama_client.logger"
        ) as mock_logger:
            client.is_available
        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args.args
        assert _URL in args
        assert 503 in args

    def test_exception_returns_false(self) -> None:
        client = OllamaClient(base_url=_URL)
        with patch(
            "src.utils.ollama_client.requests.get",
            side_effect=requests.ConnectionError("refused"),
        ):
            assert client.is_available is False

    def test_exception_logs_warning_with_type_and_message(self) -> None:
        client = OllamaClient(base_url=_URL)
        with patch(
            "src.utils.ollama_client.requests.get",
            side_effect=requests.ConnectionError("refused"),
        ), patch("src.utils.ollama_client.logger") as mock_logger:
            client.is_available
        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args.args
        assert _URL in args
        assert "ConnectionError" in args
        assert any("refused" in str(a) for a in args)
