"""EDINET の書類一覧の取得の失敗を WARNING に出し、開示なしと分ける（#1999）"""

from unittest.mock import MagicMock

import pytest
import requests

from src.market_data.adapters import edinet_client
from src.market_data.adapters.edinet_client import EdinetClient, reset_edinet_stop

_TOYOTA = {
    "secCode": "72030",
    "docTypeCode": "120",
    "docDescription": "有価証券報告書（トヨタ）",
    "filerName": "トヨタ自動車株式会社",
}


def _response(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    resp.json.return_value = body
    return resp


def _ok(*docs: dict) -> MagicMock:
    return _response({"metadata": {"status": "200"}, "results": list(docs)})


def _rate_limited() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 429
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "429 Client Error: Too Many Requests", response=resp
    )
    return resp


def _warning_texts(mock_logger: MagicMock) -> list[str]:
    texts = []
    for call in mock_logger.warning.call_args_list:
        fmt, *args = call.args
        texts.append(fmt % tuple(args) if args else fmt)
    return texts


def _day_warnings(mock_logger: MagicMock, date_str: str) -> list[str]:
    return [t for t in _warning_texts(mock_logger) if "リクエストエラー" in t and date_str in t]


@pytest.fixture(autouse=True)
def _fresh_run():
    reset_edinet_stop()
    yield
    reset_edinet_stop()


@pytest.fixture
def mock_get(monkeypatch):
    get = MagicMock()
    monkeypatch.setattr(edinet_client.requests, "get", get)
    return get


@pytest.fixture
def mock_logger(monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(edinet_client, "logger", log)
    return log


@pytest.mark.parametrize(
    "failure, reason",
    [
        (lambda: _rate_limited(), "429"),
        (lambda: requests.exceptions.Timeout("read timed out"), "Timeout"),
    ],
    ids=["http-429", "timeout"],
)
def test_request_error_warns_once_per_day_with_date_and_reason(
    mock_get, mock_logger, failure, reason
):
    """429・タイムアウトは WARNING に日付と理由を出し、同じ日を2銘柄で取っても1回だけ"""

    def _get(*_a, **_kw):
        made = failure()
        if isinstance(made, Exception):
            raise made
        return made

    mock_get.side_effect = _get

    first = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")
    second = EdinetClient().fetch_document_titles("6758", "2024-01-01", "2024-01-01")

    assert first is None and second is None
    assert mock_get.call_count == 2  # 失敗の日は持ち回らず、次の銘柄が取り直す
    warnings = _day_warnings(mock_logger, "2024-01-01")
    assert len(warnings) == 1
    assert reason in warnings[0]


def test_caller_can_tell_failed_days_from_days_without_documents(mock_get, mock_logger):
    """失敗の日は failed_days に入り、開示の無い日（404・空の一覧）は入らない"""
    by_date = {
        "2024-01-01": _ok(_TOYOTA),
        "2024-01-02": _rate_limited(),
        "2024-01-03": _response({}, status_code=404),
        "2024-01-04": _ok(),
    }
    mock_get.side_effect = lambda *_a, **kw: by_date[kw["params"]["date"]]

    client = EdinetClient()
    result = client.fetch_document_titles("7203", "2024-01-01", "2024-01-04")

    assert result == {"2024-01-01": ["有価証券報告書（トヨタ）"]}
    assert client.failed_days == ["2024-01-02"]
    summary = [t for t in _warning_texts(mock_logger) if "7203" in t]
    assert len(summary) == 1 and "1" in summary[0]


def test_all_days_failed_is_distinguishable_from_no_disclosure(mock_get, mock_logger):
    """全部の日が落ちたときも、開示の無い期間と同じ None だが failed_days で分かる"""
    mock_get.side_effect = requests.exceptions.ConnectionError("failed")
    failed = EdinetClient()
    assert failed.fetch_document_titles("7203", "2024-01-01", "2024-01-02") is None
    assert failed.failed_days == ["2024-01-01", "2024-01-02"]

    reset_edinet_stop()
    mock_get.side_effect = None
    mock_get.return_value = _ok()
    empty = EdinetClient()
    assert empty.fetch_document_titles("7203", "2024-01-01", "2024-01-02") is None
    assert empty.failed_days == []


def test_failed_days_reset_on_each_fetch(mock_get, mock_logger):
    """同じクライアントで取り直したら、前の取得の失敗の日は残らない"""
    mock_get.side_effect = [requests.exceptions.Timeout(), _ok(_TOYOTA)]
    client = EdinetClient()
    client.fetch_document_titles("7203", "2024-01-01", "2024-01-01")
    assert client.failed_days == ["2024-01-01"]

    client.fetch_document_titles("7203", "2024-01-01", "2024-01-01")
    assert client.failed_days == []


def test_ok_and_404_days_do_not_warn(mock_get, mock_logger):
    """正常な応答と 404 の日は今までどおり WARNING を出さない"""
    by_date = {"2024-01-01": _ok(_TOYOTA), "2024-01-02": _response({}, status_code=404)}
    mock_get.side_effect = lambda *_a, **kw: by_date[kw["params"]["date"]]

    client = EdinetClient()
    result = client.fetch_document_titles("7203", "2024-01-01", "2024-01-02")

    assert result == {"2024-01-01": ["有価証券報告書（トヨタ）"]}
    assert client.failed_days == []
    mock_logger.warning.assert_not_called()


def test_reset_lets_next_run_warn_again(mock_get, mock_logger):
    """回の始めの戻しの後は、同じ日の失敗をまた1回 WARNING に出す"""
    mock_get.side_effect = requests.exceptions.Timeout()
    EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")
    reset_edinet_stop()
    EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")

    assert len(_day_warnings(mock_logger, "2024-01-01")) == 2
