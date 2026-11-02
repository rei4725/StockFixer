"""market_data.symbol_master / utils.db.symbol_master のテスト（ネット・DB に出ない）。"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.market_data import symbol_master as sm
from src.market_data.types import SymbolName
from src.utils.db import symbol_master as db_sm


def _jpx_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日付": [20261001, 20261001, 20261001, 20261001],
            "コード": [7203.0, "130A", "9984", None],
            "銘柄名": [
                "トヨタ自動車",
                "Ｖｅｒｉｔａｓ　Ｉｎ　Ｓｉｌｉｃｏ",
                "ソフトバンクグループ",
                "X",
            ],
            "33業種区分": ["輸送用機器", "-", "情報・通信業", "-"],
        }
    )


def test_parse_normalizes_code_and_sector():
    records = sm.parse_jpx_listed_issues(_jpx_df())
    by_code = {r.symbol: r for r in records}
    assert set(by_code) == {"7203", "130A", "9984"}
    assert by_code["7203"].name_ja == "トヨタ自動車"
    assert by_code["7203"].sector == "輸送用機器"
    assert by_code["7203"].market == "jp"
    assert by_code["7203"].source == sm.JPX_SOURCE
    assert by_code["130A"].sector is None


def test_parse_raises_when_column_missing():
    with pytest.raises(ValueError):
        sm.parse_jpx_listed_issues(pd.DataFrame({"コード": ["7203"]}))


def test_build_reports_missing_symbols_without_network():
    downloader = MagicMock(return_value=b"xls-bytes")
    with patch.object(sm.pd, "read_excel", return_value=_jpx_df()) as read_excel, patch.object(
        sm.requests, "get"
    ) as http_get:
        result = sm.build_jp_symbol_master(["7203", "9984", "1111"], downloader=downloader)
    downloader.assert_called_once_with()
    read_excel.assert_called_once()
    http_get.assert_not_called()
    assert [r.symbol for r in result.records] == ["7203", "9984"]
    assert result.missing == ["1111"]


def test_download_uses_timeout_and_raises_for_status():
    resp = MagicMock(content=b"abc")
    with patch.object(sm.requests, "get", return_value=resp) as http_get:
        assert sm.download_jpx_listed_issues("http://example.invalid/x.xls") == b"abc"
    http_get.assert_called_once()
    assert http_get.call_args.kwargs["timeout"] > 0
    resp.raise_for_status.assert_called_once_with()


def _fake_connection(con: MagicMock):
    @contextmanager
    def _cm():
        yield con

    return _cm


def test_upsert_symbol_names_executes_per_record():
    con = MagicMock()
    records = [SymbolName("jp", "7203", "トヨタ自動車", sector="輸送用機器")]
    with patch.object(db_sm, "_db_connection", _fake_connection(con)):
        assert db_sm.upsert_symbol_names(records) == 1
    sql, params = con.execute.call_args.args
    assert "ON CONFLICT (market, symbol)" in sql
    assert params[:6] == ["jp", "7203", "トヨタ自動車", None, "輸送用機器", "jpx_listed_issues"]


def test_upsert_symbol_names_empty_skips_db():
    with patch.object(db_sm, "_db_connection") as conn:
        assert db_sm.upsert_symbol_names([]) == 0
    conn.assert_not_called()


def test_get_and_load_symbol_names():
    con = MagicMock()
    con.execute.return_value.fetchone.return_value = ("トヨタ自動車",)
    con.execute.return_value.fetchall.return_value = [("7203", "トヨタ自動車")]
    with patch.object(db_sm, "_db_connection", _fake_connection(con)):
        assert db_sm.get_symbol_name("jp", "7203") == "トヨタ自動車"
        assert db_sm.load_symbol_names("jp") == {"7203": "トヨタ自動車"}
        con.execute.return_value.fetchone.return_value = None
        assert db_sm.get_symbol_name("jp", "0000") is None


def test_parse_jpx_normalizes_fullwidth_names():
    records = sm.parse_jpx_listed_issues(_jpx_df())
    by_code = {r.symbol: r for r in records}
    assert by_code["130A"].name_ja == "Veritas In Silico"


def test_find_jpx_url_follows_link_on_page():
    page = MagicMock(text='<a href="/markets/x-att/data_j.xlsx">一覧</a>')
    with patch.object(sm.requests, "get", return_value=page):
        url = sm.find_jpx_listed_issues_url("https://www.jpx.co.jp/markets/misc/01.html")
    assert url == "https://www.jpx.co.jp/markets/x-att/data_j.xlsx"


def test_find_jpx_url_falls_back_without_link_or_page():
    with patch.object(sm.requests, "get", return_value=MagicMock(text="<html></html>")):
        assert sm.find_jpx_listed_issues_url() == sm.JPX_LISTED_ISSUES_URL
    with patch.object(sm.requests, "get", side_effect=sm.requests.ConnectionError("down")):
        assert sm.find_jpx_listed_issues_url() == sm.JPX_LISTED_ISSUES_URL


def test_download_jpx_without_url_looks_up_page_first():
    resp = MagicMock(content=b"abc")
    with patch.object(
        sm, "find_jpx_listed_issues_url", return_value="http://x.invalid/a.xlsx"
    ), patch.object(sm.requests, "get", return_value=resp) as http_get:
        assert sm.download_jpx_listed_issues() == b"abc"
    assert http_get.call_args.args[0] == "http://x.invalid/a.xlsx"


_NASDAQ_LISTED = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
    "Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "ZXZZT|NASDAQ TEST STOCK|G|Y|N|100|N|N\n"
    "File Creation Time: 1030202618:01|||||||\n"
)
_OTHER_LISTED = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "BRK.B|Berkshire Hathaway Inc. New Common Stock|N|BRK.B|N|40|N|BRK.B\n"
    "SPY|State Street SPDR S&P 500 ETF Trust|P|SPY|Y|40|N|SPY\n"
    "PNR|Pentair plc. Ordinary Share|N|PNR|N|100|N|PNR\n"
)


def test_parse_nasdaq_directory_both_layouts():
    nasdaq = {r.symbol: r for r in sm.parse_nasdaq_symbol_directory(_NASDAQ_LISTED)}
    assert set(nasdaq) == {"AAPL"}
    assert nasdaq["AAPL"].name_en == "Apple Inc."
    assert nasdaq["AAPL"].name_ja is None
    assert nasdaq["AAPL"].market == "us"
    assert nasdaq["AAPL"].source == sm.NASDAQ_SOURCE
    other = {r.symbol: r.name_en for r in sm.parse_nasdaq_symbol_directory(_OTHER_LISTED)}
    assert other == {
        "BRK-B": "Berkshire Hathaway Inc.",
        "SPY": "State Street SPDR S&P 500 ETF Trust",
        "PNR": "Pentair plc.",
    }


def test_parse_nasdaq_directory_empty_and_bad_header():
    assert sm.parse_nasdaq_symbol_directory("") == []
    with pytest.raises(ValueError):
        sm.parse_nasdaq_symbol_directory("Foo|Bar\nA|B\n")


def test_build_us_reports_missing_symbols():
    result = sm.build_us_symbol_master(
        ["AAPL", "BRK-B", "NOPE"], downloader=lambda: [_NASDAQ_LISTED, _OTHER_LISTED]
    )
    assert [r.symbol for r in result.records] == ["AAPL", "BRK-B"]
    assert result.missing == ["NOPE"]


def test_download_nasdaq_fetches_both_files():
    resp = MagicMock(text="x")
    with patch.object(sm.requests, "get", return_value=resp) as http_get:
        assert sm.download_nasdaq_symbol_directory() == ["x", "x"]
    assert [c.args[0] for c in http_get.call_args_list] == [
        sm.NASDAQ_LISTED_URL,
        sm.OTHER_LISTED_URL,
    ]


def test_refresh_us_saves_watchlist_records(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text('{"jp": ["7203"], "us": ["AAPL"]}', encoding="utf-8")
    record = SymbolName("us", "AAPL", None, name_en="Apple Inc.", source=sm.NASDAQ_SOURCE)
    built = sm.SymbolMasterBuildResult(records=[record])
    with patch(
        "src.utils.data_path_utils.get_watchlist_path", return_value=str(path)
    ), patch.object(sm, "build_us_symbol_master", return_value=built) as build, patch.object(
        sm, "upsert_symbol_names", return_value=1
    ) as upsert:
        assert sm.refresh_us_symbol_master() == 1
    build.assert_called_once_with(["AAPL"])
    upsert.assert_called_once_with([record])


def test_symbol_label_uses_cached_names():
    names = {"7203": "トヨタ自動車", "8725": "MS&ADインシュアランスグループホールディングス"}
    with patch.object(db_sm, "load_symbol_display_names", return_value=names) as load:
        assert db_sm.symbol_label("jp", "7203") == "7203 トヨタ自動車"
        assert db_sm.symbol_label(None, "7203.T") == "7203.T トヨタ自動車"
        assert db_sm.symbol_name_suffix("jp", "9999") == ""
        long_name = db_sm.symbol_display_name("jp", "8725")
    assert long_name is not None and len(long_name) == 24 and long_name.endswith("…")
    load.assert_called_once_with("jp")


def test_symbol_label_falls_back_to_code_when_db_fails():
    with patch.object(db_sm, "load_symbol_display_names", side_effect=RuntimeError("db down")):
        assert db_sm.symbol_label("us", "AAPL") == "AAPL"
        assert db_sm.symbol_name_suffix("us", "AAPL") == ""


def test_guess_market():
    assert db_sm.guess_market("7203") == "jp"
    assert db_sm.guess_market("130A") == "jp"
    assert db_sm.guess_market("6758.T") == "jp"
    assert db_sm.guess_market("AAPL") == "us"
    assert db_sm.guess_market("BRK-B") == "us"


def test_load_symbol_display_names_prefers_japanese():
    con = MagicMock()
    con.execute.return_value.fetchall.return_value = [("AAPL", "Apple Inc.")]
    with patch.object(db_sm, "_db_connection", _fake_connection(con)):
        assert db_sm.load_symbol_display_names("us") == {"AAPL": "Apple Inc."}
    assert "COALESCE(name_ja, name_en)" in con.execute.call_args.args[0]
