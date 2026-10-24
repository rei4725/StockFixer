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
