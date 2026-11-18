"""バックテストの最終日の強制決済（Backtester.simulate_trading から切り出した）。"""

from typing import Any

import pandas as pd

from src.backtest.execution import ExecutionModel


def liquidate_at_end(
    execution: ExecutionModel,
    df: pd.DataFrame,
    close_col: str,
    exec_avg_vol: pd.Series,
    enable_short: bool,
    cash: float,
    cash_gross: float,
    position: int,
    short_position: int,
    short_price: float,
    short_entry_vol: float,
    trade_log: list[dict[str, Any]],
    short_trade_log: list[dict[str, Any]],
) -> tuple[float, float]:
    """最終日に残ったロングとショートを決済し、trade_log と short_trade_log に行を足す。

    Returns:
        決済の後の (cash, cash_gross)
    """
    # 最終日に未決済ポジションを強制決済。execution_lag > 0 では最後の足の出来高を
    # 先読みしないよう、前の足までの平均出来高（mark-to-market と同じ値）を使う（#1347）
    final_vol = float(exec_avg_vol.iloc[-1]) if len(exec_avg_vol) else 0.0
    if position > 0:
        price = df.iloc[-1][close_col]
        proceeds = execution.sell_proceeds(position, price, final_vol)
        proceeds_gross = position * price
        cash += proceeds
        cash_gross += proceeds_gross
        trade_log.append(
            {
                "date": df.index[-1],
                "action": "final_sell",
                "price": price,
                "qty": position,
                "cash": cash,
                "cash_gross": cash_gross,
            }
        )

    # 最終日に未決済ショートを強制返済
    if enable_short and short_position > 0:
        price = df.iloc[-1][close_col]
        # エントリー売り: short_price * qty * (1 - fee - slip)（エントリー時点の出来高）
        # カバー買い: price * qty * (1 + fee + slip)
        net_pnl = execution.sell_proceeds(
            short_position, short_price, short_entry_vol
        ) - execution.buy_cost(short_position, price, final_vol)
        pnl = (short_price - price) * short_position
        cash += net_pnl
        cash_gross += pnl
        short_trade_log.append(
            {
                "entry_price": short_price,
                "exit_price": price,
                "qty": short_position,
                "pnl": net_pnl,
            }
        )
        trade_log.append(
            {
                "date": df.index[-1],
                "action": "final_short_cover",
                "price": price,
                "qty": short_position,
                "cash": cash,
                "cash_gross": cash_gross,
            }
        )
    return cash, cash_gross
