"""#2201: 日次で earnings_calendar に決算日を書く道のテスト"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from src.market_data.event_calendar import refresh_event_calendars
from src.orchestration.jobs.earnings_calendar_stage import refresh_earnings_calendar_stage

_MOD = "src.market_data.event_calendar"


def _task(market: str, symbol: str) -> SimpleNamespace:
    return SimpleNamespace(market=market, symbol=symbol)


@patch(f"{_MOD}.bulk_upsert")
@patch(f"{_MOD}._db_connection")
@patch(f"{_MOD}._fetch_event_dates_from_yfinance")
def test_refresh_writes_rows_for_jp_symbols(mock_fetch, mock_conn, mock_upsert):
    mock_conn.return_value.__enter__.return_value = MagicMock()
    mock_fetch.return_value = pd.DatetimeIndex(["2026-11-10", "2027-02-10"])

    saved = refresh_event_calendars([_task("jp", "7203"), _task("us", "AAPL")])

    assert saved == 1
    mock_fetch.assert_called_once_with("jp", "7203")
    df = mock_upsert.call_args.args[2]
    assert mock_upsert.call_args.args[1] == "earnings_calendar"
    assert list(df["symbol"]) == ["7203", "7203"]
    assert list(df["event_date"]) == ["2026-11-10", "2027-02-10"]


@patch(f"{_MOD}.bulk_upsert")
@patch(f"{_MOD}._db_connection")
@patch(f"{_MOD}._fetch_event_dates_from_yfinance")
def test_refresh_dedupes_symbols(mock_fetch, mock_conn, mock_upsert):
    mock_conn.return_value.__enter__.return_value = MagicMock()
    mock_fetch.return_value = pd.DatetimeIndex(["2026-11-10"])

    saved = refresh_event_calendars([_task("jp", "7203"), _task("jp", "7203")])

    assert saved == 1
    assert mock_fetch.call_count == 1


@patch(f"{_MOD}._save_event_dates")
@patch(f"{_MOD}._fetch_event_dates_from_yfinance")
def test_refresh_warns_when_fetch_empty(mock_fetch, mock_save):
    mock_fetch.return_value = pd.DatetimeIndex([])

    with patch(f"{_MOD}.logger") as mock_logger:
        saved = refresh_event_calendars([_task("jp", "7203")])

    assert saved == 0
    mock_save.assert_not_called()
    assert mock_logger.warning.called


@patch(f"{_MOD}.bulk_upsert", side_effect=RuntimeError("db down"))
@patch(f"{_MOD}._db_connection")
@patch(f"{_MOD}._fetch_event_dates_from_yfinance")
def test_refresh_warns_when_write_fails(mock_fetch, mock_conn, mock_upsert):
    mock_conn.return_value.__enter__.return_value = MagicMock()
    mock_fetch.return_value = pd.DatetimeIndex(["2026-11-10"])

    with patch(f"{_MOD}.logger") as mock_logger:
        saved = refresh_event_calendars([_task("jp", "7203")])

    assert saved == 0
    messages = [str(c.args[0]) for c in mock_logger.warning.call_args_list]
    assert any("保存失敗" in m for m in messages)
    assert any("書けなかった" in m for m in messages)


def test_stage_logs_warning_and_does_not_raise():
    with (
        patch(
            "src.market_data.event_calendar.refresh_event_calendars",
            side_effect=RuntimeError("boom"),
        ),
        patch("src.orchestration.jobs.earnings_calendar_stage.logger") as mock_logger,
    ):
        refresh_earnings_calendar_stage([_task("jp", "7203")])

    assert mock_logger.warning.called


def test_stage_calls_refresh_with_tasks():
    tasks = [_task("jp", "7203")]
    with patch(
        "src.market_data.event_calendar.refresh_event_calendars", return_value=1
    ) as mock_refresh:
        refresh_earnings_calendar_stage(tasks)

    mock_refresh.assert_called_once_with(tasks)
