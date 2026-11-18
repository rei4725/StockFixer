"""liquidate_at_end（最終日の強制決済）の単体テスト。

期待値は、Backtester.simulate_trading から移す前の式をそのまま書いたもの。
"""

from typing import Any

import pandas as pd
import pytest

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.final_liquidation import liquidate_at_end

_FEE = 0.001
_SLIP = 0.0005


def _execution() -> ExecutionModel:
    return ExecutionModel(TradingCosts(fee_rate=_FEE, slippage_rate=_SLIP, lot_size=1))


def _df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    return pd.DataFrame({"Close": [100.0, 101.0, 105.0]}, index=idx)


def _exec_avg_vol() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    return pd.Series([1000.0, 2000.0, 3000.0], index=idx)


def _run(
    position: int, short_position: int, enable_short: bool
) -> tuple[float, float, list[dict[str, Any]], list[dict[str, Any]]]:
    trade_log: list[dict[str, Any]] = []
    short_trade_log: list[dict[str, Any]] = []
    cash, cash_gross = liquidate_at_end(
        _execution(),
        _df(),
        "Close",
        _exec_avg_vol(),
        enable_short,
        500_000.0,
        510_000.0,
        position,
        short_position,
        110.0,
        800.0,
        trade_log,
        short_trade_log,
    )
    return cash, cash_gross, trade_log, short_trade_log


def test_long_only_final_sell() -> None:
    cash, cash_gross, log, short_log = _run(position=100, short_position=0, enable_short=True)
    ex = _execution()
    exp_cash = 500_000.0 + ex.sell_proceeds(100, 105.0, 3000.0)
    exp_gross = 510_000.0 + 100 * 105.0
    assert cash == pytest.approx(exp_cash)
    assert cash_gross == pytest.approx(exp_gross)
    assert short_log == []
    assert len(log) == 1
    row = log[0]
    assert list(row) == ["date", "action", "price", "qty", "cash", "cash_gross"]
    assert row["date"] == _df().index[-1]
    assert row["action"] == "final_sell"
    assert row["price"] == 105.0
    assert row["qty"] == 100
    assert row["cash"] == pytest.approx(exp_cash)
    assert row["cash_gross"] == pytest.approx(exp_gross)


def test_short_only_final_short_cover() -> None:
    cash, cash_gross, log, short_log = _run(position=0, short_position=50, enable_short=True)
    ex = _execution()
    net_pnl = ex.sell_proceeds(50, 110.0, 800.0) - ex.buy_cost(50, 105.0, 3000.0)
    exp_cash = 500_000.0 + net_pnl
    exp_gross = 510_000.0 + (110.0 - 105.0) * 50
    assert cash == pytest.approx(exp_cash)
    assert cash_gross == pytest.approx(exp_gross)
    assert short_log == [
        {"entry_price": 110.0, "exit_price": 105.0, "qty": 50, "pnl": pytest.approx(net_pnl)}
    ]
    assert len(log) == 1
    row = log[0]
    assert list(row) == ["date", "action", "price", "qty", "cash", "cash_gross"]
    assert row["action"] == "final_short_cover"
    assert row["price"] == 105.0
    assert row["qty"] == 50
    assert row["cash"] == pytest.approx(exp_cash)
    assert row["cash_gross"] == pytest.approx(exp_gross)


def test_no_positions_leaves_cash_and_logs() -> None:
    cash, cash_gross, log, short_log = _run(position=0, short_position=0, enable_short=True)
    assert cash == 500_000.0
    assert cash_gross == 510_000.0
    assert log == []
    assert short_log == []


def test_short_left_open_when_short_disabled() -> None:
    cash, cash_gross, log, short_log = _run(position=0, short_position=50, enable_short=False)
    assert cash == 500_000.0
    assert cash_gross == 510_000.0
    assert log == []
    assert short_log == []


def test_empty_avg_vol_uses_zero_volume() -> None:
    trade_log: list[dict[str, Any]] = []
    cash, _ = liquidate_at_end(
        _execution(),
        _df(),
        "Close",
        pd.Series([], dtype=float),
        False,
        0.0,
        0.0,
        10,
        0,
        0.0,
        0.0,
        trade_log,
        [],
    )
    assert cash == pytest.approx(_execution().sell_proceeds(10, 105.0, 0.0))
