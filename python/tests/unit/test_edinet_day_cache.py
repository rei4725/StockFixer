"""EDINET の1日分の一覧を、1回のバッチの中で銘柄の間で使い回す（#1692）"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
import requests

from src.market_data import pipeline
from src.market_data.adapters import edinet_client
from src.market_data.adapters.edinet_client import EdinetClient, reset_edinet_stop

_TOYOTA = {
    "secCode": "72030",
    "docTypeCode": "120",
    "docDescription": "有価証券報告書（トヨタ）",
    "filerName": "トヨタ自動車株式会社",
}
_SONY = {
    "secCode": "67580",
    "docTypeCode": "160",
    "docDescription": "",
    "filerName": "ソニーグループ株式会社",
}
_NOT_TARGET = {
    "secCode": "72030",
    "docTypeCode": "999",
    "docDescription": "対象外の書類",
    "filerName": "トヨタ自動車株式会社",
}
_RATE_LIMITED = {"StatusCode": 429, "message": "Too Many Requests"}


def _response(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    resp.json.return_value = body
    return resp


def _ok(*docs: dict) -> MagicMock:
    return _response({"metadata": {"status": "200"}, "results": list(docs)})


def _dates_called(mock_get: MagicMock) -> list[str]:
    return [call.kwargs["params"]["date"] for call in mock_get.call_args_list]


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


def test_each_day_is_fetched_once_across_symbols(mock_get):
    """銘柄が変わっても日ごとに1回だけ呼び、銘柄ごとの絞り込みの結果は今までと同じ"""
    by_date = {
        "2024-01-01": _ok(_TOYOTA, _NOT_TARGET),
        "2024-01-02": _ok(_SONY),
        "2024-01-03": _ok(_TOYOTA, _SONY),
    }
    mock_get.side_effect = lambda *_a, **kw: by_date[kw["params"]["date"]]

    toyota = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-03")
    sony = EdinetClient().fetch_document_titles("6758", "2024-01-01", "2024-01-03")
    nothing = EdinetClient().fetch_document_titles("9984", "2024-01-01", "2024-01-03")

    assert toyota == {
        "2024-01-01": ["有価証券報告書（トヨタ）"],
        "2024-01-03": ["有価証券報告書（トヨタ）"],
    }
    assert sony == {
        "2024-01-02": ["ソニーグループ株式会社"],
        "2024-01-03": ["ソニーグループ株式会社"],
    }
    assert nothing is None
    assert sorted(_dates_called(mock_get)) == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_longer_range_fetches_only_the_missing_days(mock_get):
    """期間の違う銘柄では、まだ取っていない日だけを呼ぶ"""
    mock_get.return_value = _ok(_TOYOTA)

    EdinetClient().fetch_document_titles("7203", "2024-01-02", "2024-01-03")
    result = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-03")

    assert result is not None and sorted(result) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert _dates_called(mock_get) == ["2024-01-02", "2024-01-03", "2024-01-01"]


def test_threads_fetching_the_same_day_call_once(mock_get):
    """並列の銘柄が同じ日を同時に取っても1回だけ呼び、全員が同じ一覧を読む"""
    workers = 8
    barrier = threading.Barrier(workers)

    def _slow_get(*_a, **_kw):
        time.sleep(0.05)
        return _ok(_TOYOTA, _SONY)

    def _fetch(symbol: str):
        barrier.wait(timeout=5)
        return EdinetClient().fetch_document_titles(symbol, "2024-01-01", "2024-01-01")

    mock_get.side_effect = _slow_get
    symbols = ["7203", "6758"] * (workers // 2)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_fetch, symbols))

    assert mock_get.call_count == 1
    assert results == [
        {"2024-01-01": ["有価証券報告書（トヨタ）"]},
        {"2024-01-01": ["ソニーグループ株式会社"]},
    ] * (workers // 2)


def test_fetching_one_day_does_not_block_another_day(mock_get):
    """取得中の日があっても、別の日の取得は待たされない"""
    second_day_called = threading.Event()
    first_day_saw_second = []

    def _get(*_a, **kw):
        if kw["params"]["date"] == "2024-01-01":
            first_day_saw_second.append(second_day_called.wait(timeout=5))
        else:
            second_day_called.set()
        return _ok(_TOYOTA)

    mock_get.side_effect = _get
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(EdinetClient()._fetch_day, "2024-01-01", "72030")
        second = executor.submit(EdinetClient()._fetch_day, "2024-01-02", "72030")
        assert first.result(timeout=10) == ["有価証券報告書（トヨタ）"]
        assert second.result(timeout=10) == ["有価証券報告書（トヨタ）"]

    assert first_day_saw_second == [True]


def test_reset_discards_days_of_the_previous_run(mock_get):
    """回の始めの戻しの後は新しく取り直し、前の回の一覧を読まない"""
    mock_get.return_value = _ok(_TOYOTA)
    EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")

    reset_edinet_stop()
    mock_get.return_value = _ok(_SONY)
    result = EdinetClient().fetch_document_titles("6758", "2024-01-01", "2024-01-01")

    assert mock_get.call_count == 2
    assert result == {"2024-01-01": ["ソニーグループ株式会社"]}


def test_batch_pipeline_discards_days_at_start(mock_get):
    """バッチの回の始めで、持ち回りの一覧も捨てる"""
    mock_get.return_value = _ok(_TOYOTA)
    EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")

    pipeline.run_batch_pipeline([])
    EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")

    assert mock_get.call_count == 2


@pytest.mark.parametrize(
    "failure",
    [
        requests.exceptions.ConnectionError("failed"),
        requests.exceptions.Timeout(),
        requests.exceptions.HTTPError("500"),
        ValueError("予期しない例外"),
    ],
    ids=["connection", "timeout", "http-error", "unexpected"],
)
def test_failed_day_is_not_carried_over(mock_get, failure):
    """失敗した日の空の結果は持ち回らず、次の銘柄が取り直す"""
    mock_get.side_effect = [failure, _ok(_TOYOTA, _SONY)]

    first = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")
    second = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")
    third = EdinetClient().fetch_document_titles("6758", "2024-01-01", "2024-01-01")

    assert first is None
    assert second == {"2024-01-01": ["有価証券報告書（トヨタ）"]}
    assert third == {"2024-01-01": ["ソニーグループ株式会社"]}
    assert mock_get.call_count == 2


def test_unsuccessful_status_body_stops_as_before_and_is_not_carried_over(mock_get, monkeypatch):
    """本文の StatusCode が成功でない日は打ち切りを立て、戻した後の回は取り直す"""
    mock_logger = MagicMock()
    monkeypatch.setattr(edinet_client, "logger", mock_logger)
    mock_get.return_value = _response(_RATE_LIMITED)

    results = [
        EdinetClient().fetch_document_titles(symbol, "2024-01-01", "2024-01-03")
        for symbol in ("7203", "6758")
    ]

    assert results == [None, None]
    assert mock_get.call_count == 1
    assert mock_logger.warning.call_count == 1

    reset_edinet_stop()
    mock_get.return_value = _ok(_TOYOTA)
    result = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")

    assert result == {"2024-01-01": ["有価証券報告書（トヨタ）"]}
    assert mock_get.call_count == 2


def test_stop_keeps_returning_none_even_for_days_already_held(mock_get):
    """打ち切りの後の銘柄は、持っている日があっても今までどおり None を返す"""
    by_date = {"2024-01-01": _ok(_TOYOTA), "2024-01-02": _response(_RATE_LIMITED)}
    mock_get.side_effect = lambda *_a, **kw: by_date[kw["params"]["date"]]

    first = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-02")
    second = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")

    assert first == {"2024-01-01": ["有価証券報告書（トヨタ）"]}
    assert second is None
    assert mock_get.call_count == 2


def test_waiting_thread_does_not_fetch_after_stop(mock_get):
    """同じ日を待っていたスレッドは、打ち切りが立ったら取りに行かない"""
    in_request = threading.Event()
    release = threading.Event()

    def _get(*_a, **_kw):
        in_request.set()
        assert release.wait(timeout=5)
        return _response(_RATE_LIMITED)

    mock_get.side_effect = _get
    with ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(EdinetClient()._fetch_day, "2024-01-01", "72030")
        assert in_request.wait(timeout=5)
        waiter = executor.submit(EdinetClient()._fetch_day, "2024-01-01", "67580")
        time.sleep(0.05)
        release.set()
        assert holder.result(timeout=10) == []
        assert waiter.result(timeout=10) == []

    assert mock_get.call_count == 1


def test_404_day_is_carried_over_as_a_day_without_documents(mock_get):
    """404 は今までどおり開示の無い日で、銘柄ごとに呼び直さない"""
    mock_get.return_value = _response({}, status_code=404)

    results = [
        EdinetClient().fetch_document_titles(symbol, "2024-01-01", "2024-01-01")
        for symbol in ("7203", "6758")
    ]

    assert results == [None, None]
    assert mock_get.call_count == 1


def test_body_without_results_is_not_carried_over(mock_get):
    """results の無い本文は成功と言えないので、今までどおり次の銘柄が取り直す"""
    mock_get.side_effect = [_response({"metadata": {"status": "400"}}), _ok(_TOYOTA)]

    first = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")
    second = EdinetClient().fetch_document_titles("7203", "2024-01-01", "2024-01-01")

    assert first is None
    assert second == {"2024-01-01": ["有価証券報告書（トヨタ）"]}


def test_returned_titles_are_not_shared_between_callers(mock_get):
    """返した一覧を呼び手が変えても、次の銘柄の結果は変わらない"""
    mock_get.return_value = _ok(_TOYOTA)

    client = EdinetClient()
    client._fetch_day("2024-01-01", "72030").append("呼び手が足した行")

    assert client._fetch_day("2024-01-01", "72030") == ["有価証券報告書（トヨタ）"]
