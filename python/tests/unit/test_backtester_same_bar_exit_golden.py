"""Unit Test: Backtester.simulate_trading の同じ足の損切り・利確の出力を固定する（#1915）

execution_lag=0 で損切りと利確の両方に当たる決まった入力について、result_df と
metrics を、まとめる前のコードで取った値と突き合わせる。
"""

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.backtest.backtester import Backtester

_EXPECTED_TRADES = [
    ("2024-01-01", "buy", 100, 9900, 9010.000000000116, 10000),
    ("2024-01-03", "stop_loss", 94, 9900, 938679.4000000001, 940600),
    ("2024-01-04", "buy", 95, 9800, 6748.400000000256, 9600),
    ("2024-01-09", "take_profit", 106, 9800, 1044509.6000000002, 1048400),
    ("2024-01-10", "buy", 104, 10000, 3469.600000000326, 8400),
    ("2024-01-15", "sell", 103, 10000, 1032439.6000000003, 1038400),
]

_EXPECTED_METRICS: dict[str, Any] = {
    "final_cash": 1032439.6,
    "total_return": 0.03244,
    "num_trades": 3,
    "win_rate": 0.3333,
    "profit_factor": 1.442,
    "sharpe_ratio": 1.076,
    "sharpe_per_trade": 0.125892,
    "max_drawdown": -0.061322,
    "avg_position_fraction": 0.993493,
    "min_position_fraction": 0.99099,
    "max_position_fraction": 0.996678,
    "avg_position_value": 987000.0,
    "atr_fallback_trades": 0,
    "avg_win": 0.113674,
    "avg_loss": 0.036773,
    "trade_returns": [0.11367368421052639, -0.061939999999999856, -0.01160576923076912],
    "gross_final_cash": 1038400,
    "gross_total_return": 0.0384,
    "gross_sharpe_ratio": 1.2736,
    "gross_max_drawdown": -0.0594,
    "cost_impact_cash": 5960.4,
    "cost_impact_return": 0.00596,
    "short_return": 0.0,
    "short_num_trades": 0,
}


def _run() -> tuple[pd.DataFrame, dict[str, Any]]:
    index = pd.date_range("2024-01-01", periods=12, freq="B")
    df = pd.DataFrame(
        {
            "Close": [100, 98, 94, 95, 99, 103, 106, 104, 102, 101, 103, 105],
            "Volume": [10_000] * 12,
        },
        index=index,
    )
    # 0: 買い → 2: 損切り（-6%）→ 3: 買い → 6: 利確（+11.6%）→ 7: 買い → 10: 売り
    signal = pd.Series([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0], index=index)
    bt = Backtester(
        model_manager=MagicMock(),
        signal_generator=MagicMock(),
        data_loader=MagicMock(),
        start_date=None,
        end_date=None,
        market="jp",
        symbol="7203",
        initial_cash=1_000_000,
        fee_rate=0.001,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        execution_lag=0,
    )
    return bt.simulate_trading(df, signal)


def test_same_bar_exit_result_df_is_fixed():
    result_df, _ = _run()
    assert len(result_df) == len(_EXPECTED_TRADES)
    for row, (date, action, price, qty, cash, cash_gross) in zip(
        result_df.to_dict("records"), _EXPECTED_TRADES
    ):
        assert row["date"] == pd.Timestamp(date)
        assert row["action"] == action
        assert row["price"] == price
        assert row["qty"] == qty
        assert row["cash"] == pytest.approx(cash, rel=1e-12)
        assert row["cash_gross"] == pytest.approx(cash_gross, rel=1e-12)


def test_same_bar_exit_trade_log_keys_are_fixed():
    result_df, _ = _run()
    exits = result_df[result_df["action"].isin(["stop_loss", "take_profit"])]
    assert list(exits["action"]) == ["stop_loss", "take_profit"]
    assert exits[["position_fraction", "sizing_mode"]].isna().all().all()


def test_same_bar_exit_metrics_are_fixed():
    _, metrics = _run()
    assert set(metrics) == set(_EXPECTED_METRICS)
    for key, expected in _EXPECTED_METRICS.items():
        assert metrics[key] == pytest.approx(expected, rel=1e-9), key
