"""market_data.fundamentals と db.stock_fundamentals のユニットテスト。

- yf.Ticker をモックして .info / 財務 DataFrame を差し込み、パースを検証する
- 取得失敗（例外・空）時に None を返しログを出すこと
- 派生指標（売上 CAGR・マージン）の計算が正しいこと
- upsert → load のラウンドトリップ（一時 DuckDB ファイル使用）
"""

import glob
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.market_data.fundamentals import _revenue_cagr_3y, fetch_fundamentals
from src.market_data.types import FundamentalRecord
from src.utils.db.stock_fundamentals import (
    load_all_fundamentals,
    load_fundamentals,
    upsert_fundamentals,
)


def _financials_df():
    """売上・営業利益・純利益を含む財務 DataFrame（列＝期間、新しい順）。"""
    periods = pd.to_datetime(["2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"])
    return pd.DataFrame(
        {
            periods[0]: [400.0, 100.0, 80.0],
            periods[1]: [300.0, 70.0, 55.0],
            periods[2]: [200.0, 40.0, 30.0],
            periods[3]: [100.0, 20.0, 15.0],
        },
        index=["Total Revenue", "Operating Income", "Net Income"],
    )


def _make_ticker(info=None, financials=None, income_stmt=None, balance_sheet=None):
    mock = MagicMock()
    mock.info = info if info is not None else {}
    fin = financials if financials is not None else _financials_df()
    mock.financials = fin
    mock.income_stmt = income_stmt if income_stmt is not None else fin
    mock.balance_sheet = (
        balance_sheet
        if balance_sheet is not None
        else pd.DataFrame({pd.Timestamp("2024-12-31"): [50.0]}, index=["Cash And Cash Equivalents"])
    )
    return mock


class TestFetchFundamentals(unittest.TestCase):
    def test_parse_from_info(self):
        info = {
            "totalRevenue": 400.0,
            "netIncomeToCommon": 80.0,
            "trailingEps": 5.5,
            "returnOnEquity": 0.25,
            "operatingMargins": 0.25,
            "profitMargins": 0.20,
            "debtToEquity": 1.5,
            "totalCash": 123.0,
            "marketCap": 9999.0,
            "sharesOutstanding": 1000.0,
        }
        with patch("src.market_data.fundamentals.yf.Ticker", return_value=_make_ticker(info)):
            rec = fetch_fundamentals("us", "AAPL")

        self.assertIsInstance(rec, FundamentalRecord)
        self.assertEqual(rec.market, "us")
        self.assertEqual(rec.symbol, "AAPL")
        self.assertEqual(rec.revenue, 400.0)
        self.assertEqual(rec.net_income, 80.0)
        self.assertEqual(rec.eps, 5.5)
        self.assertEqual(rec.roe, 0.25)
        self.assertEqual(rec.op_margin, 0.25)
        self.assertEqual(rec.net_margin, 0.20)
        self.assertEqual(rec.debt_to_equity, 1.5)
        self.assertEqual(rec.cash, 123.0)
        self.assertEqual(rec.market_cap, 9999.0)
        self.assertEqual(rec.shares_outstanding, 1000.0)
        # income_stmt（=financials）から営業利益
        self.assertEqual(rec.operating_income, 100.0)

    def test_fallback_to_financials_and_derived_margins(self):
        """info にマージン等が無い場合、財務 DataFrame から算出する。"""
        info = {"marketCap": 1.0}  # 最低限のデータでフォールバック誘発
        with patch("src.market_data.fundamentals.yf.Ticker", return_value=_make_ticker(info)):
            rec = fetch_fundamentals("us", "MSFT")

        self.assertIsNotNone(rec)
        # 財務 DataFrame の直近列から取得
        self.assertEqual(rec.revenue, 400.0)
        self.assertEqual(rec.operating_income, 100.0)
        self.assertEqual(rec.net_income, 80.0)
        # 派生マージン: 100/400, 80/400
        self.assertAlmostEqual(rec.op_margin, 0.25)
        self.assertAlmostEqual(rec.net_margin, 0.20)
        # 現金は balance_sheet から
        self.assertEqual(rec.cash, 50.0)

    def test_revenue_cagr(self):
        """売上 CAGR: 100 → 400 を3年で = (4)^(1/3) - 1 ≈ 0.587。"""
        with patch(
            "src.market_data.fundamentals.yf.Ticker", return_value=_make_ticker({"marketCap": 1.0})
        ):
            rec = fetch_fundamentals("us", "X")
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.revenue_cagr_3y, (4.0) ** (1.0 / 3.0) - 1.0, places=6)

    def test_cagr_helper_insufficient_data(self):
        single = pd.DataFrame({pd.Timestamp("2024-12-31"): [100.0]}, index=["Total Revenue"])
        self.assertIsNone(_revenue_cagr_3y(single))
        self.assertIsNone(_revenue_cagr_3y(None))
        self.assertIsNone(_revenue_cagr_3y(pd.DataFrame()))

    def test_exception_returns_none(self):
        with patch("src.market_data.fundamentals.yf.Ticker", side_effect=RuntimeError("boom")):
            with self.assertLogs("src.market_data.fundamentals", level="WARNING") as cm:
                rec = fetch_fundamentals("us", "FAIL")
        self.assertIsNone(rec)
        self.assertTrue(any("取得失敗" in m for m in cm.output))

    def test_empty_data_returns_none(self):
        empty_ticker = _make_ticker(
            info={},
            financials=pd.DataFrame(),
            income_stmt=pd.DataFrame(),
            balance_sheet=pd.DataFrame(),
        )
        with patch("src.market_data.fundamentals.yf.Ticker", return_value=empty_ticker):
            with self.assertLogs("src.market_data.fundamentals", level="WARNING") as cm:
                rec = fetch_fundamentals("us", "EMPTY")
        self.assertIsNone(rec)
        self.assertTrue(any("有効な財務指標なし" in m for m in cm.output))


class TestFundamentalsDb(unittest.TestCase):
    def setUp(self):
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        for path in glob.glob(self.tmp_db + "*"):
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _record(self, symbol="AAPL", revenue=400.0):
        return FundamentalRecord(
            market="us",
            symbol=symbol,
            as_of=pd.Timestamp("2026-06-04 00:00:00").to_pydatetime(),
            revenue=revenue,
            operating_income=100.0,
            net_income=80.0,
            eps=5.5,
            roe=0.25,
            op_margin=0.25,
            net_margin=0.20,
            debt_to_equity=1.5,
            cash=50.0,
            market_cap=9999.0,
            shares_outstanding=1000.0,
            revenue_cagr_3y=0.5,
        )

    def test_upsert_load_roundtrip(self):
        upsert_fundamentals(self._record())
        loaded = load_fundamentals("us", "AAPL")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["revenue"], 400.0)
        self.assertEqual(loaded["eps"], 5.5)
        self.assertEqual(loaded["symbol"], "AAPL")

    def test_upsert_is_idempotent(self):
        upsert_fundamentals(self._record(revenue=100.0))
        upsert_fundamentals(self._record(revenue=500.0))
        loaded = load_fundamentals("us", "AAPL")
        self.assertEqual(loaded["revenue"], 500.0)
        df = load_all_fundamentals()
        self.assertEqual(len(df[df["symbol"] == "AAPL"]), 1)

    def test_load_missing_returns_none(self):
        self.assertIsNone(load_fundamentals("us", "NOPE"))

    def test_load_all(self):
        upsert_fundamentals(self._record(symbol="AAPL"))
        upsert_fundamentals(self._record(symbol="MSFT"))
        df = load_all_fundamentals()
        self.assertEqual(len(df), 2)
        self.assertSetEqual(set(df["symbol"]), {"AAPL", "MSFT"})


if __name__ == "__main__":
    unittest.main()
