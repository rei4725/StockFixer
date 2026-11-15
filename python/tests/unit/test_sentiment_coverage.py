"""sentiment_coverage（市場ごとのセンチメントの埋まり具合の診断）のテスト。"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd

from src.market_data.sentiment_coverage import load_sentiment_rows, sentiment_coverage


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": ["jp", "jp", "jp", "jp", "jp", "us", "us"],
            "symbol": ["7203", "7203", "6758", "6758", "9984", "AAPL", "AAPL"],
            "date": pd.to_datetime(
                [
                    "2026-10-01",
                    "2026-10-02",
                    "2026-10-02",
                    "2026-10-03",
                    "2026-10-04",
                    "2026-10-01",
                    "2026-10-02",
                ]
            ),
            "news_count_lag1": [0, 2, 1, 3, 1, 0, 0],
            "sentiment_score_lag1": [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
        }
    )


class TestSentimentCoverage:
    def test_counts_per_market(self) -> None:
        out = sentiment_coverage(_rows()).set_index("market")

        jp = out.loc["jp"]
        assert jp["rows"] == 5
        assert jp["news_rows"] == 4
        assert jp["news_days"] == 3
        assert jp["zero_score_rows"] == 3
        assert jp["zero_score_rate"] == 0.75
        assert jp["last_nonzero_date"] == pd.Timestamp("2026-10-02")

        us = out.loc["us"]
        assert us["rows"] == 2
        assert us["news_rows"] == 0
        assert us["news_days"] == 0
        assert us["zero_score_rows"] == 0
        assert pd.isna(us["zero_score_rate"])
        assert pd.isna(us["last_nonzero_date"])

    def test_markets_are_sorted(self) -> None:
        out = sentiment_coverage(_rows().iloc[::-1])
        assert list(out["market"]) == ["jp", "us"]

    def test_since_filters_rows(self) -> None:
        out = sentiment_coverage(_rows(), since="2026-10-03").set_index("market")
        assert out.loc["jp", "rows"] == 2
        assert out.loc["jp", "news_rows"] == 2
        assert out.loc["jp", "zero_score_rate"] == 1.0
        assert pd.isna(out.loc["jp", "last_nonzero_date"])
        assert "us" not in out.index

    def test_missing_values_are_not_news_nor_nonzero(self) -> None:
        df = pd.DataFrame(
            {
                "market": ["jp", "jp"],
                "date": pd.to_datetime(["2026-10-01", "2026-10-02"]),
                "news_count_lag1": [float("nan"), 1.0],
                "sentiment_score_lag1": [float("nan"), float("nan")],
            }
        )
        out = sentiment_coverage(df).set_index("market")
        assert out.loc["jp", "news_rows"] == 1
        # 点の無い行は 0 点として数えず、0 でない点としても数えない
        assert out.loc["jp", "zero_score_rows"] == 0
        assert pd.isna(out.loc["jp", "last_nonzero_date"])

    def test_custom_columns(self) -> None:
        df = _rows().rename(
            columns={"news_count_lag1": "news_count", "sentiment_score_lag1": "sentiment_score"}
        )
        out = sentiment_coverage(df, score_col="sentiment_score", news_col="news_count")
        assert out.set_index("market").loc["jp", "zero_score_rows"] == 3

    def test_empty_input(self) -> None:
        out = sentiment_coverage(_rows().iloc[0:0])
        assert out.empty
        assert "zero_score_rate" in out.columns

    def test_does_not_mutate_input(self) -> None:
        df = _rows()
        before = df.copy()
        sentiment_coverage(df, since="2026-10-02")
        pd.testing.assert_frame_equal(df, before)


class TestLoadSentimentRows:
    def test_reads_selected_columns(self) -> None:
        con = MagicMock()
        con.execute.return_value.fetchall.return_value = [
            ("jp", "7203", "2026-10-01", 1, 0.0),
        ]

        @contextmanager
        def _fake():
            yield con

        with patch("src.market_data.sentiment_coverage._db_connection", _fake):
            df = load_sentiment_rows(since="2026-10-01")

        sql, params = con.execute.call_args.args
        assert "stock_features" in sql
        assert params == ["2026-10-01"]
        assert list(df.columns) == [
            "market",
            "symbol",
            "date",
            "news_count_lag1",
            "sentiment_score_lag1",
        ]
        assert df.loc[0, "date"] == pd.Timestamp("2026-10-01")

    def test_without_since_has_no_params(self) -> None:
        con = MagicMock()
        con.execute.return_value.fetchall.return_value = []

        @contextmanager
        def _fake():
            yield con

        with patch("src.market_data.sentiment_coverage._db_connection", _fake):
            df = load_sentiment_rows()

        sql, params = con.execute.call_args.args
        assert "WHERE" not in sql
        assert params == []
        assert df.empty
