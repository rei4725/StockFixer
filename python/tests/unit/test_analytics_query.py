"""ユニットテスト: AnalyticsQuery ポートと 2 実装。"""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd

from src.domain.ports import AnalyticsQuery
from src.infrastructure.in_memory import InMemoryAnalyticsQuery
from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery

_EMPTY = {
    "tracked_count": 0,
    "comparable_count": 0,
    "avg_paper_slippage": 0.0,
    "avg_real_slippage": 0.0,
    "avg_abs_price_diff": 0.0,
    "avg_abs_diff_ratio": 0.0,
    "max_abs_price_diff": 0.0,
}


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConnection:
    def __init__(self, row):
        self.calls: list[tuple[str, list]] = []
        self._row = row

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return _FakeCursor(self._row)


def _patched_connection(con):
    @contextmanager
    def _cm():
        yield con

    return _cm


class TestPortContract(unittest.TestCase):
    def test_port_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            AnalyticsQuery()  # type: ignore[abstract]

    def test_both_implementations_satisfy_port(self):
        self.assertIsInstance(InMemoryAnalyticsQuery(), AnalyticsQuery)
        self.assertIsInstance(PostgresAnalyticsQuery(), AnalyticsQuery)


class TestInMemoryAnalyticsQuery(unittest.TestCase):
    def test_defaults_to_zeros(self):
        self.assertEqual(InMemoryAnalyticsQuery().paper_real_diff_summary(), _EMPTY)

    def test_returns_seeded_value(self):
        seeded = dict(_EMPTY, tracked_count=5, avg_paper_slippage=0.01)
        q = InMemoryAnalyticsQuery(paper_real_diff=seeded)
        self.assertEqual(q.paper_real_diff_summary(recent_days=30), seeded)


class TestPostgresAnalyticsQuery(unittest.TestCase):
    def test_maps_row_to_summary_dict(self):
        row = (10, 7, 0.001, 0.002, 3.5, 0.0035, 9.0)
        con = _FakeConnection(row)
        with patch(
            "src.infrastructure.persistence.analytics_query.db_connection",
            _patched_connection(con),
        ):
            result = PostgresAnalyticsQuery().paper_real_diff_summary(recent_days=7)

        self.assertEqual(result["tracked_count"], 10)
        self.assertEqual(result["comparable_count"], 7)
        self.assertAlmostEqual(result["avg_paper_slippage"], 0.001)
        self.assertAlmostEqual(result["avg_real_slippage"], 0.002)
        self.assertAlmostEqual(result["avg_abs_price_diff"], 3.5)
        self.assertAlmostEqual(result["avg_abs_diff_ratio"], 0.0035)
        self.assertAlmostEqual(result["max_abs_price_diff"], 9.0)

    def test_null_row_becomes_zeros(self):
        con = _FakeConnection((None, None, None, None, None, None, None))
        with patch(
            "src.infrastructure.persistence.analytics_query.db_connection",
            _patched_connection(con),
        ):
            result = PostgresAnalyticsQuery().paper_real_diff_summary()

        self.assertEqual(result, _EMPTY)

    def test_queries_paper_real_diff_table(self):
        con = _FakeConnection((0, 0, None, None, None, None, None))
        with patch(
            "src.infrastructure.persistence.analytics_query.db_connection",
            _patched_connection(con),
        ):
            PostgresAnalyticsQuery().paper_real_diff_summary(recent_days=14)

        sql, params = con.calls[0]
        self.assertIn("FROM paper_real_diff", sql)
        self.assertIn("FILTER", sql)
        self.assertEqual(len(params), 1)


class _DuckDBConnection:
    """集計の SQL を実際に流すため、Postgres の %s を DuckDB の ? に置き換えて渡す。"""

    def __init__(self, con):
        self._con = con

    def execute(self, sql, params=None):
        return self._con.execute(sql.replace("%s", "?"), params)


class TestPostgresAnalyticsQuerySlippageSign(unittest.TestCase):
    """スリッページは損の向きを正にして平均する（#1261）。

    side は BUY=1・SELL=2・SHORT=3・SHORT_COVER=4。
    """

    def _summary(self, rows):
        import duckdb

        con = duckdb.connect(":memory:")
        con.execute(
            "CREATE TABLE paper_real_diff ("
            "side INTEGER, signal_price DOUBLE, paper_price DOUBLE, real_price DOUBLE, "
            "paper_slippage DOUBLE, real_slippage DOUBLE, price_diff DOUBLE, "
            "paper_filled_at TIMESTAMP, real_checked_at TIMESTAMP, created_at TIMESTAMP)"
        )
        for side, slip in rows:
            con.execute(
                "INSERT INTO paper_real_diff VALUES "
                "(?, 100.0, ?, ?, ?, ?, 0.0, NULL, NULL, CURRENT_TIMESTAMP)",
                [side, 100.0 * (1 + slip), 100.0 * (1 + slip), slip, slip],
            )
        with patch(
            "src.infrastructure.persistence.analytics_query.db_connection",
            _patched_connection(_DuckDBConnection(con)),
        ):
            return PostgresAnalyticsQuery().paper_real_diff_summary(recent_days=7)

    def test_buy_high_and_sell_low_both_count_as_cost(self):
        result = self._summary([(1, 0.01), (2, -0.01)])
        self.assertAlmostEqual(result["avg_paper_slippage"], 0.01)
        self.assertAlmostEqual(result["avg_real_slippage"], 0.01)

    def test_buy_high_and_sell_high_cancel_out(self):
        result = self._summary([(1, 0.01), (2, 0.01)])
        self.assertAlmostEqual(result["avg_paper_slippage"], 0.0)
        self.assertAlmostEqual(result["avg_real_slippage"], 0.0)

    def test_short_low_and_short_cover_high_count_as_cost(self):
        result = self._summary([(3, -0.01), (4, 0.01)])
        self.assertAlmostEqual(result["avg_paper_slippage"], 0.01)
        self.assertAlmostEqual(result["avg_real_slippage"], 0.01)


class TestPostgresAnalyticsQueryAccuracyDelegation(unittest.TestCase):
    """予測精度系 3 メソッドは prediction BC 所有の SQL へ引数をそのまま委譲する。"""

    _MOD = "src.prediction.db.accuracy"

    def test_drift_summary_delegates(self):
        sentinel = pd.DataFrame({"x": [1]})
        with patch(f"{self._MOD}.load_drift_summary", return_value=sentinel) as mock_load:
            result = PostgresAnalyticsQuery().drift_summary(
                horizon=3, recent_n=20, model_name="challenger"
            )
        mock_load.assert_called_once_with(horizon=3, recent_n=20, model_name="challenger")
        self.assertIs(result, sentinel)

    def test_prediction_accuracy_delegates(self):
        sentinel = pd.DataFrame({"x": [1]})
        with patch(f"{self._MOD}.load_prediction_accuracy", return_value=sentinel) as mock_load:
            result = PostgresAnalyticsQuery().prediction_accuracy(
                market="jp", symbol="7203", horizon=5, limit=10
            )
        mock_load.assert_called_once_with(market="jp", symbol="7203", horizon=5, limit=10)
        self.assertIs(result, sentinel)

    def test_weekly_accuracy_snapshots_delegates(self):
        sentinel = pd.DataFrame({"x": [1]})
        with patch(
            f"{self._MOD}.load_weekly_accuracy_snapshots", return_value=sentinel
        ) as mock_load:
            result = PostgresAnalyticsQuery().weekly_accuracy_snapshots(n_weeks=8)
        mock_load.assert_called_once_with(n_weeks=8)
        self.assertIs(result, sentinel)


class TestInMemoryAnalyticsQueryAccuracy(unittest.TestCase):
    def test_defaults_to_empty_frames(self):
        q = InMemoryAnalyticsQuery()
        self.assertTrue(q.drift_summary().empty)
        self.assertTrue(q.prediction_accuracy().empty)
        self.assertTrue(q.weekly_accuracy_snapshots().empty)

    def test_returns_copy_of_seeded_frame(self):
        """呼び出し側が返り値を書き換えても仕込んだ値は汚れない。"""
        seeded = pd.DataFrame({"direction_match": [True]})
        q = InMemoryAnalyticsQuery(accuracy=seeded)
        q.prediction_accuracy()["direction_match"] = False
        self.assertTrue(q.prediction_accuracy()["direction_match"].iloc[0])


if __name__ == "__main__":
    unittest.main()
