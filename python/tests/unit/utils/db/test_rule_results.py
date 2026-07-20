"""rule_results.py の実DB回帰テスト（rule_best_by_symbol / rule_daily_signals）。

DuckDB版からPostgres(psycopg3, %s プレースホルダ)への移行時に自動テストが
一件も無く、手元の使い捨てスクリプトでのみ検証されていたための追加分。
呼び出し元（rule_selector.py / rule_engine/pipeline.py）は広い try/except で
本モジュールの例外を握りつぶすため、回帰があっても「ログに出て握りつぶされる」
だけで気づけない。ここではDELETE→INSERTによる上書き（upsert相当）が
正しく「1行だけ・最新値で」残ることを中心に検証する。
"""

import pandas as pd
import pytest

from src.utils.db._connection import _db_connection
from src.utils.db.rule_results import (
    load_effective_rules,
    load_rule_best_all,
    load_rule_signals_by_date,
    upsert_rule_best,
    upsert_rule_signal,
)


def _make_best(market="us", symbol="TEST", **overrides):
    kwargs = dict(
        market=market,
        symbol=symbol,
        best_rule="golden_cross",
        win_rate=0.6,
        net_profit=1000.0,
        num_trades=10,
        profit_factor=1.5,
        max_drawdown=-0.2,
        backtest_start="2024-01-01",
        backtest_end="2024-06-30",
    )
    kwargs.update(overrides)
    upsert_rule_best(**kwargs)


def _make_signal(signal_date="2024-06-01", market="us", symbol="TEST", **overrides):
    kwargs = dict(
        signal_date=signal_date,
        market=market,
        symbol=symbol,
        rule_name="golden_cross",
        signal=1,
        price=150.0,
    )
    kwargs.update(overrides)
    upsert_rule_signal(**kwargs)


class TestUpsertRuleBest:
    def test_save_and_reload_roundtrip(self):
        _make_best(symbol="ROUND", best_rule="rsi_reversal", win_rate=0.7, net_profit=500.0)

        with _db_connection() as con:
            row = con.execute(
                """
                SELECT market, symbol, best_rule, win_rate, net_profit, num_trades,
                       profit_factor, max_drawdown, backtest_start, backtest_end
                FROM rule_best_by_symbol WHERE market = %s AND symbol = %s
                """,
                ["us", "ROUND"],
            ).fetchone()

        assert row is not None
        assert row[0] == "us"
        assert row[1] == "ROUND"
        assert row[2] == "rsi_reversal"
        assert row[3] == pytest.approx(0.7)
        assert row[4] == pytest.approx(500.0)
        assert row[5] == 10

    def test_second_upsert_for_same_key_replaces_not_duplicates(self):
        _make_best(symbol="DUP", best_rule="golden_cross", win_rate=0.4, net_profit=100.0)
        _make_best(symbol="DUP", best_rule="macd_cross", win_rate=0.9, net_profit=999.0)

        with _db_connection() as con:
            rows = con.execute(
                "SELECT best_rule, win_rate, net_profit FROM rule_best_by_symbol "
                "WHERE market = %s AND symbol = %s",
                ["us", "DUP"],
            ).fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "macd_cross"
        assert rows[0][1] == pytest.approx(0.9)
        assert rows[0][2] == pytest.approx(999.0)

    def test_different_symbols_do_not_collide(self):
        _make_best(symbol="A1", best_rule="rule_a")
        _make_best(symbol="A2", best_rule="rule_b")

        with _db_connection() as con:
            rows = con.execute(
                "SELECT symbol, best_rule FROM rule_best_by_symbol "
                "WHERE market = %s AND symbol IN (%s, %s)",
                ["us", "A1", "A2"],
            ).fetchall()

        by_symbol = {r[0]: r[1] for r in rows}
        assert by_symbol == {"A1": "rule_a", "A2": "rule_b"}


class TestLoadRuleBestAll:
    def test_load_rule_best_all_returns_saved_row(self):
        _make_best(symbol="LOADALL", best_rule="golden_cross", win_rate=0.55)

        df = load_rule_best_all("us")

        assert isinstance(df, pd.DataFrame)
        matched = df[df["symbol"] == "LOADALL"]
        assert len(matched) == 1
        assert matched.iloc[0]["best_rule"] == "golden_cross"

    def test_load_rule_best_all_filters_by_market(self):
        _make_best(market="us", symbol="USONLY")
        _make_best(market="jp", symbol="JPONLY")

        df_us = load_rule_best_all("us")
        df_jp = load_rule_best_all("jp")

        assert "USONLY" in set(df_us["symbol"])
        assert "USONLY" not in set(df_jp["symbol"])
        assert "JPONLY" in set(df_jp["symbol"])
        assert "JPONLY" not in set(df_us["symbol"])


class TestLoadEffectiveRules:
    def test_filters_by_win_rate_and_net_profit_threshold(self):
        _make_best(symbol="GOOD", win_rate=0.8, net_profit=500.0)
        _make_best(symbol="LOWWIN", win_rate=0.1, net_profit=500.0)
        _make_best(symbol="NOPROFIT", win_rate=0.8, net_profit=-10.0)

        df = load_effective_rules("us", min_win_rate=0.5, min_net_profit=0.0)
        symbols = set(df["symbol"])

        assert "GOOD" in symbols
        assert "LOWWIN" not in symbols
        assert "NOPROFIT" not in symbols

    def test_default_thresholds_exclude_zero_profit(self):
        _make_best(symbol="ZEROPROFIT", win_rate=0.9, net_profit=0.0)

        df = load_effective_rules("us")

        assert "ZEROPROFIT" not in set(df["symbol"])


class TestUpsertRuleSignal:
    def test_save_and_reload_roundtrip(self):
        _make_signal(
            signal_date="2024-06-01",
            symbol="SIGROUND",
            rule_name="golden_cross",
            signal=1,
            price=123.45,
        )

        with _db_connection() as con:
            row = con.execute(
                """
                SELECT signal_date, market, symbol, rule_name, signal, price
                FROM rule_daily_signals
                WHERE signal_date = %s AND market = %s AND symbol = %s
                """,
                ["2024-06-01", "us", "SIGROUND"],
            ).fetchone()

        assert row is not None
        assert row[2] == "SIGROUND"
        assert row[3] == "golden_cross"
        assert row[4] == 1
        assert row[5] == pytest.approx(123.45)

    def test_second_upsert_for_same_key_replaces_not_duplicates(self):
        _make_signal(
            signal_date="2024-06-02", symbol="SIGDUP", rule_name="rule_a", signal=1, price=10.0
        )
        _make_signal(
            signal_date="2024-06-02", symbol="SIGDUP", rule_name="rule_b", signal=-1, price=20.0
        )

        with _db_connection() as con:
            rows = con.execute(
                "SELECT rule_name, signal, price FROM rule_daily_signals "
                "WHERE signal_date = %s AND market = %s AND symbol = %s",
                ["2024-06-02", "us", "SIGDUP"],
            ).fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "rule_b"
        assert rows[0][1] == -1
        assert rows[0][2] == pytest.approx(20.0)


class TestLoadRuleSignalsByDate:
    def test_roundtrip_returns_saved_signal(self):
        _make_signal(signal_date="2024-06-03", symbol="DATELOAD", signal=1)

        df = load_rule_signals_by_date("2024-06-03", "us")

        assert "DATELOAD" in set(df["symbol"])

    def test_filters_out_other_dates(self):
        _make_signal(signal_date="2024-06-04", symbol="DAY1", signal=1)
        _make_signal(signal_date="2024-06-05", symbol="DAY2", signal=1)

        df_day1 = load_rule_signals_by_date("2024-06-04", "us")
        df_day2 = load_rule_signals_by_date("2024-06-05", "us")

        assert "DAY1" in set(df_day1["symbol"])
        assert "DAY2" not in set(df_day1["symbol"])
        assert "DAY2" in set(df_day2["symbol"])
        assert "DAY1" not in set(df_day2["symbol"])

    def test_filters_by_market(self):
        _make_signal(signal_date="2024-06-06", market="us", symbol="USONLY", signal=1)
        _make_signal(signal_date="2024-06-06", market="jp", symbol="JPONLY", signal=1)

        df_us = load_rule_signals_by_date("2024-06-06", "us")
        df_jp = load_rule_signals_by_date("2024-06-06", "jp")

        assert "USONLY" in set(df_us["symbol"])
        assert "USONLY" not in set(df_jp["symbol"])
        assert "JPONLY" in set(df_jp["symbol"])
