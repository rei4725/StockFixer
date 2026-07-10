"""src/utils/heartbeat.py（healthchecks.io 死活監視 ping #496）のユニットテスト"""

import os
import sys
from unittest.mock import MagicMock, patch

import requests

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.settings import settings  # noqa: E402
from src.utils.heartbeat import SCHEDULER_ALIVE_SLUG, ping_heartbeat  # noqa: E402


def _enable(monkeypatch, key: str = "test-ping-key", base_url: str = "https://hc-ping.com"):
    monkeypatch.setattr(settings, "HEALTHCHECKS_PING_KEY", key)
    monkeypatch.setattr(settings, "HEALTHCHECKS_BASE_URL", base_url)


def test_ping_is_noop_without_key(monkeypatch):
    """キー未設定（空文字）の場合は HTTP リクエストを送らず False を返すこと"""
    monkeypatch.setattr(settings, "HEALTHCHECKS_PING_KEY", "")

    with patch("src.utils.heartbeat.requests.get") as mock_get:
        assert ping_heartbeat("daily_pipeline") is False

    mock_get.assert_not_called()


def test_ping_success_sends_to_slug_url(monkeypatch):
    """成功 ping は <base>/<key>/<slug> へ create=1 付きで送られること"""
    _enable(monkeypatch)

    with patch("src.utils.heartbeat.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        assert ping_heartbeat("daily_pipeline") is True

    mock_get.assert_called_once_with(
        "https://hc-ping.com/test-ping-key/daily_pipeline",
        params={"create": "1"},
        timeout=10,
    )


def test_ping_failure_uses_fail_endpoint(monkeypatch):
    """success=False の場合は /fail エンドポイントへ送られること"""
    _enable(monkeypatch)

    with patch("src.utils.heartbeat.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        assert ping_heartbeat("weekly_model_training", success=False) is True

    called_url = mock_get.call_args[0][0]
    assert called_url == "https://hc-ping.com/test-ping-key/weekly_model_training/fail"


def test_ping_swallows_request_exception(monkeypatch):
    """接続エラーでも例外を送出せず False を返すこと（ジョブ本体を落とさない）"""
    _enable(monkeypatch)

    with patch("src.utils.heartbeat.requests.get") as mock_get:
        mock_get.side_effect = requests.ConnectionError("network down")
        assert ping_heartbeat("daily_pipeline") is False


def test_ping_non_2xx_returns_false(monkeypatch):
    """非 2xx 応答は False を返すこと"""
    _enable(monkeypatch)

    with patch("src.utils.heartbeat.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=503)
        assert ping_heartbeat("daily_pipeline") is False


def test_scheduler_alive_slug_constant():
    """recovery_poller が使う slug 定数が期待値であること"""
    assert SCHEDULER_ALIVE_SLUG == "scheduler-alive"
