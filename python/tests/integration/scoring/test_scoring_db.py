"""src.scoring.db の統合テスト（テスト DB、テストごとにロールバック）。"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.scoring import db as sdb
from src.scoring.version import read_app_version
from src.utils.db._connection import _db_connection
from src.utils.db.rule_results import ensure_rule_tables, upsert_rule_signal

D = date(2026, 11, 2)


def _count(table: str) -> int:
    with _db_connection() as con:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # nosec B608


def _put_prediction(predicted_at: str, price: float = 100.0) -> None:
    with _db_connection() as con:
        con.execute("DELETE FROM prediction_results")
        con.execute(
            "INSERT INTO prediction_results"
            " (market, symbol, predicted_at, model_version, current_price,"
            "  diff_ratio, diff_ratio_5d) VALUES ('jp', '7203', %s, 'v1', %s, 0.01, 0.02)",
            [predicted_at, price],
        )


def test_log_rule_signal_versions_idempotent():
    ensure_rule_tables()
    sdb.ensure_scoring_tables()
    upsert_rule_signal(D, "jp", "7203", "r1", 1, 100.0)
    assert sdb.log_rule_signal_versions(D, "jp", "2.37.0") == 1
    assert sdb.log_rule_signal_versions(D, "jp", "2.38.0") == 0
    assert _count("rule_signal_versions") == 1


def test_snapshot_predictions_keeps_history():
    sdb.ensure_scoring_tables()
    _put_prediction("20261102_224526")
    assert sdb.snapshot_predictions("2.37.0") == 1
    _put_prediction("20261103_224526", 101.0)
    assert sdb.snapshot_predictions("2.37.0") == 1
    df = sdb.load_prediction_history("jp", date(2026, 11, 1))
    assert sorted(df["predicted_at"]) == ["20261102_224526", "20261103_224526"]


def test_upsert_rule_outcomes_twice_same_rows():
    sdb.ensure_scoring_tables()
    row = {
        "signal_date": D,
        "market": "jp",
        "symbol": "7203",
        "rule_name": "r1",
        "signal": 1,
        "horizon": 1,
        "hit": True,
        "baseline": 0.5,
        "signed_return": 0.01,
        "app_version": "2.37.0",
        "scored_at": datetime(2026, 11, 3, 1, 0, 0),
    }
    assert sdb.upsert_rule_outcomes(pd.DataFrame([row])) == 1
    row2 = {**row, "signed_return": 0.05, "hit": False}
    sdb.upsert_rule_outcomes(pd.DataFrame([row2]))
    assert _count("rule_signal_outcomes") == 1
    with _db_connection() as con:
        r = con.execute("SELECT signed_return, hit FROM rule_signal_outcomes").fetchone()
    assert r == (0.05, False)


def test_load_rule_signals_joins_version():
    ensure_rule_tables()
    sdb.ensure_scoring_tables()
    upsert_rule_signal(date(2026, 10, 30), "jp", "7203", "r1", 1, 100.0)
    upsert_rule_signal(D, "jp", "7203", "r1", -1, 101.0)
    with _db_connection() as con:
        con.execute(
            "INSERT INTO rule_signal_versions VALUES (%s, 'jp', '7203', '2.37.0', now())", [D]
        )
    df = sdb.load_rule_signals("jp", date(2026, 10, 1)).sort_values("signal_date")
    assert df["app_version"].tolist()[0] is None
    assert df["app_version"].tolist()[1] == "2.37.0"


def test_load_bars_filters_market_and_timeframe():
    with _db_connection() as con:
        for mk, tf in [("jp", "1d"), ("jp", "1h"), ("us", "1d")]:
            con.execute(
                "INSERT INTO market_data_raw"
                " (market, symbol, ticker, timeframe, ts, close, adj_close)"
                " VALUES (%s, 'ZZ1', 'ZZ1', %s, '2026-11-02', 10, 9)",
                [mk, tf],
            )
    df = sdb.load_bars("jp", date(2026, 11, 1))
    assert list(df.columns) == ["symbol", "ts", "close", "adj_close"]
    assert len(df[df["symbol"] == "ZZ1"]) == 1


def test_load_prediction_signals_h1_and_rule_best():
    ensure_rule_tables()
    sdb.ensure_scoring_tables()
    with _db_connection() as con:
        for h in (1, 3):
            con.execute(
                "INSERT INTO prediction_accuracy (market, symbol, model_name, predicted_at,"
                " horizon, predicted_ratio)"
                " VALUES ('jp', '7203', 'm', '20261102_224526', %s, 0.01)",
                [h],
            )
        con.execute(
            "INSERT INTO rule_best_by_symbol (market, symbol, best_rule, win_rate, net_profit,"
            " num_trades, backtest_start, backtest_end) VALUES"
            " ('jp', '7203', 'r1', 0.6, 1.0, 10, '2026-01-01', '2026-10-01')"
        )
    h1 = sdb.load_prediction_signals_h1("jp", date(2026, 11, 1))
    assert len(h1) == 1
    assert list(h1.columns) == ["symbol", "model_name", "predicted_at", "predicted_ratio"]
    assert len(sdb.load_rule_best("jp")) == 1
    assert sdb.snapshot_rule_best(D, "jp") == 1
    assert sdb.snapshot_rule_best(D, "jp") == 1
    assert len(sdb.load_rule_best_history("jp")) == 1


def test_read_app_version_strips_bom(tmp_path: Path):
    p = tmp_path / "VERSION"
    p.write_bytes(b"\xef\xbb\xbf2.37.0\n")
    assert read_app_version(p) == "2.37.0"
    assert read_app_version(tmp_path / "missing") is None
