"""ユニットテスト: order_run_summary を期間で読む関数と、日・市場ごとの流れの表（#1981）。"""

from datetime import datetime

import pandas as pd

from src.infrastructure.persistence.order_run_repository import load_order_run_summaries
from src.reporting.order_run_trend import (
    OrderRunRecord,
    build_order_run_trend,
    format_order_run_csv,
    format_order_run_markdown,
)
from src.utils.db._connection import _db_connection


def _insert(run_id, market, mode, run_at, buy, sell, short, skipped, skipped_min, turnover):
    with _db_connection() as con:
        con.execute(
            "INSERT INTO order_run_summary "
            "(run_id, market, mode, run_at, buy_orders, sell_orders, short_orders, "
            "skipped, skipped_min_change, total_turnover, min_change_ratio) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [run_id, market, mode, run_at, buy, sell, short, skipped, skipped_min, turnover, 0.005],
        )


def _seed():
    # 同じ日・市場に2回、別の日に1回、同じ日に別の市場で1回
    _insert("r1", "jp", "paper", datetime(2026, 11, 16, 23, 50), 2, 1, 0, 3, 1, 100000.0)
    _insert("r2", "jp", "paper", datetime(2026, 11, 16, 23, 55), 1, 0, 1, 2, 0, 50000.5)
    _insert("r3", "jp", "paper", datetime(2026, 11, 17, 23, 50), 0, 2, 0, 1, 1, 30000.0)
    _insert("r4", "us", "paper", datetime(2026, 11, 16, 14, 30), 4, 0, 0, 0, 0, 7000.0)


def _record(run_id, market, run_at, buy=0, sell=0, short=0, skipped=0, turnover=0.0, mode="paper"):
    return OrderRunRecord(
        run_id=run_id,
        market=market,
        mode=mode,
        run_at=run_at,
        buy_orders=buy,
        sell_orders=sell,
        short_orders=short,
        skipped=skipped,
        skipped_min_change=0,
        total_turnover=turnover,
        min_change_ratio=0.005,
    )


class TestLoadOrderRunSummaries:
    def test_reads_all_rows_in_run_at_order(self):
        _seed()
        records = load_order_run_summaries()
        assert [r.run_id for r in records] == ["r4", "r1", "r2", "r3"]
        r1 = records[1]
        assert r1.market == "jp"
        assert r1.mode == "paper"
        assert (r1.buy_orders, r1.sell_orders, r1.short_orders) == (2, 1, 0)
        assert (r1.skipped, r1.skipped_min_change) == (3, 1)
        assert r1.total_turnover == 100000.0
        assert r1.min_change_ratio == 0.005
        assert r1.run_at == datetime(2026, 11, 16, 23, 50)

    def test_filters_by_since_until_and_market(self):
        _seed()
        records = load_order_run_summaries(
            since=datetime(2026, 11, 16), until=datetime(2026, 11, 17), market="jp"
        )
        assert [r.run_id for r in records] == ["r1", "r2"]

    def test_empty_table_returns_empty_list(self):
        assert load_order_run_summaries() == []


class TestBuildOrderRunTrend:
    def test_sums_per_date_and_market_from_db(self):
        _seed()
        df = build_order_run_trend(load_order_run_summaries())

        assert list(df[["date", "market"]].itertuples(index=False, name=None)) == [
            ("2026-11-16", "jp"),
            ("2026-11-16", "us"),
            ("2026-11-17", "jp"),
        ]
        jp16 = df.iloc[0]
        assert jp16["runs"] == 2
        assert jp16["buy_orders"] == 3
        assert jp16["sell_orders"] == 1
        assert jp16["short_orders"] == 1
        assert jp16["skipped"] == 5
        assert jp16["skipped_min_change"] == 1
        assert jp16["total_turnover"] == 150000.5
        assert jp16["mode"] == "paper"
        assert df.iloc[1]["runs"] == 1
        assert df.iloc[1]["total_turnover"] == 7000.0
        assert df.iloc[2]["runs"] == 1
        assert df.iloc[2]["sell_orders"] == 2
        assert df.iloc[2]["total_turnover"] == 30000.0

    def test_mode_values_are_copied_as_is(self):
        df = build_order_run_trend(
            [
                _record("a", "jp", datetime(2026, 11, 16, 1), mode="paper"),
                _record("b", "jp", datetime(2026, 11, 16, 2), mode="other"),
            ]
        )
        assert len(df) == 1
        assert df.iloc[0]["runs"] == 2
        assert df.iloc[0]["mode"] == "other,paper"

    def test_empty_records_returns_empty_frame(self):
        df = build_order_run_trend([])
        assert df.empty
        assert "total_turnover" in df.columns
        assert "runs" in df.columns
        assert format_order_run_markdown(df) == "データなし"

    def test_tz_aware_run_at_uses_utc_date(self):
        run_at = pd.Timestamp("2026-11-17 08:50", tz="Asia/Tokyo").to_pydatetime()
        df = build_order_run_trend([_record("a", "jp", run_at, turnover=1.0)])
        assert df.iloc[0]["date"] == "2026-11-16"


class TestFormat:
    def test_markdown_and_csv(self):
        df = build_order_run_trend(
            [_record("a", "jp", datetime(2026, 11, 16, 1), buy=1, turnover=1234.5)]
        )
        md = format_order_run_markdown(df)
        assert md.splitlines()[0].startswith("| date | market | mode | runs |")
        assert "1234.5" in md
        csv = format_order_run_csv(df)
        assert csv.splitlines()[0].startswith("date,market,mode,runs,")
