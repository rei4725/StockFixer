"""紙の口座の元手 1000 万円への入金の移行と、入金を利益に数えない週の成績のテスト（#1877）。"""

import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.trading.brokers.base import OrderSide
from src.trading.paper_cash_events import replay_paper_cash
from src.trading.paper_equity import compute_weekly_results
from src.trading.paper_weekly_backfill import backfill_previous_weeks
from src.utils.db import migration_runner

_MIGRATIONS = Path(migration_runner.__file__).parent / "migrations"
_MIGRATION = "0010_paper_balance_10m_postgres.sql"


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("jp", "7203", int(OrderSide.BUY), 100, 3000.0, "2026-04-02"),
            ("jp", "6758", int(OrderSide.BUY), 100, 2000.0, "2026-04-03"),
            ("jp", "7203", int(OrderSide.SELL), 200, 3100.0, "2026-04-10"),
        ],
        columns=["market", "symbol", "side", "qty", "fill_price", "filled_at"],
    )


def _migration_sql() -> str:
    return (_MIGRATIONS / _MIGRATION).read_text(encoding="utf-8")


def test_replayed_cash_from_10m_matches_balance_after_migration() -> None:
    orders = _orders()
    # 100 万円の口座の今の残高（売りは持ち分の 100 株まで）
    balance_1m = replay_paper_cash(orders, 1_000_000.0)
    assert balance_1m == 1_000_000.0 - 300_000.0 - 200_000.0 + 310_000.0
    # 移行の SQL が足す額
    sql = _migration_sql()
    assert "UPDATE paper_balance SET balance = balance + 9000000.0" in sql
    after_migration = balance_1m + 9_000_000.0
    # 投資家バトルと同じく、最初の約定から元手 1000 万円でたどり直した現金
    assert abs(replay_paper_cash(orders, 10_000_000.0) - after_migration) < 1e-6


def test_migration_records_deposit_and_does_not_run_twice() -> None:
    sql = _migration_sql()
    assert "INSERT INTO paper_cash_events" in sql
    applied = {
        p.name.split("_", 1)[0] for p in _MIGRATIONS.glob("*_postgres.sql") if p.name != _MIGRATION
    }
    con = MagicMock()
    con.execute.return_value.fetchall.return_value = [(v,) for v in applied]
    assert migration_runner.run_migrations(con) == 1
    # 2回目は版の記録に 0010 があるので流さない
    con2 = MagicMock()
    con2.execute.return_value.fetchall.return_value = [(v,) for v in applied | {"0010"}]
    assert migration_runner.run_migrations(con2) == 0
    statements = [str(c.args[0]) for c in con2.execute.call_args_list]
    assert not any("paper_balance" in s for s in statements)


def _flat_prices(market: str, symbol: str) -> pd.Series:
    return pd.Series(1000.0, index=pd.bdate_range("2026-03-01", "2026-12-31"))


def _no_benchmark(ticker: str, start: str, end: str) -> dict:
    return {"total_return": 0.0, "start": start}


def test_switch_week_return_does_not_jump_with_deposit() -> None:
    orders = pd.DataFrame(
        [("jp", "7203", int(OrderSide.BUY), 100, 1000.0, "2026-11-02")],
        columns=["market", "symbol", "side", "qty", "fill_price", "filled_at"],
    )
    deposits = [(pd.Timestamp("2026-11-18"), 9_000_000.0)]
    # 2026-11-16〜20 の週（入金のある週）
    result = compute_weekly_results(
        orders, _flat_prices, 1_000_000.0, date(2026, 11, 21), _no_benchmark, deposits=deposits
    )
    assert result["markets"]["jp"]["weekly_return"] == 0.0
    assert result["initial_balance"] == 10_000_000.0
    # 入金の前の週は元手 100 万円のまま
    before = compute_weekly_results(
        orders, _flat_prices, 1_000_000.0, date(2026, 11, 14), _no_benchmark, deposits=deposits
    )
    assert before["initial_balance"] == 1_000_000.0
    assert before["markets"]["jp"]["weekly_return"] == 0.0


def test_backfill_does_not_rewrite_weeks_before_switch(tmp_path: Path) -> None:
    # initial_balance の無い古い形の、欠けのある切り替えの前の週のファイル
    old = {
        "week": "2026-W45",
        "start": "2026-11-02",
        "end": "2026-11-06",
        "markets": {"jp": {"weekly_return": 0.01, "excess_return": None}},
    }
    path = tmp_path / "2026-W45.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    compute_week = MagicMock()
    backfill_previous_weeks(
        date(2026, 11, 20), compute_week, str(tmp_path), frozen_before=date(2026, 11, 16)
    )
    compute_week.assert_not_called()
    assert json.loads(path.read_text(encoding="utf-8")) == old
    assert os.listdir(tmp_path) == ["2026-W45.json"]
