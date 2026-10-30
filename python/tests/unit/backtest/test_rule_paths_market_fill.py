"""#1418: 本番のルールの発注は成行（trading/rule_execution.py の OrderType.MARKET）。

rule_backtester と戦略ファクトリーのバックテストは、出来高の少ない銘柄でも指値の判定を
当てず、翌バーの始値で約定する。予想のシグナルの道（既定）は指値の判定が当たる。
"""

from __future__ import annotations

import pandas as pd

from config.settings import LIMIT_ORDER_AVG_VOLUME_THRESHOLD
from src.backtest import factory, rule_backtester
from src.backtest.backtester import Backtester

_LOW_VOLUME = LIMIT_ORDER_AVG_VOLUME_THRESHOLD / 50


def _gap_up_df() -> pd.DataFrame:
    """1000 円で動かない5本の後、翌バーが 1050 円に窓を開ける（指値 1001 円に届かない）。"""
    bars = [(1000.0, 1000.0, 1000.0, 1000.0)] * 5 + [
        (1050.0, 1060.0, 1040.0, 1050.0),
        (1050.0, 1050.0, 1050.0, 1050.0),
    ]
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="B")
    return pd.DataFrame(
        {
            "Open": [b[0] for b in bars],
            "High": [b[1] for b in bars],
            "Low": [b[2] for b in bars],
            "Close": [b[3] for b in bars],
            "Volume": [_LOW_VOLUME] * len(bars),
        },
        index=idx,
    )


def _buy_rows(bt: Backtester) -> pd.DataFrame:
    df = _gap_up_df()
    signal = pd.Series([1 if i == 4 else 0 for i in range(len(df))], index=df.index)
    result_df, _ = bt.simulate_trading(df, signal)
    if result_df.empty:
        return result_df
    return result_df[result_df["action"] == "buy"]


def test_rule_backtester_fills_at_next_open_on_low_volume():
    bt = rule_backtester._make_backtester("jp", 1_000_000, 0.0, 0.0, None, None, False)
    buys = _buy_rows(bt)
    assert len(buys) == 1
    assert buys.iloc[0]["price"] == 1050.0


def test_factory_fills_at_next_open_on_low_volume():
    bt = factory._make_backtester("jp", 1_000_000, 0.0, 0.0, None, False)
    buys = _buy_rows(bt)
    assert len(buys) == 1
    assert buys.iloc[0]["price"] == 1050.0


def test_default_backtester_keeps_limit_fill():
    bt = Backtester(
        model_manager=None,
        signal_generator=None,
        data_loader=None,
        start_date=None,
        end_date=None,
        market="jp",
        symbol="7203",
        fee_rate=0.0,
        slippage=0.0,
    )
    assert _buy_rows(bt).empty
