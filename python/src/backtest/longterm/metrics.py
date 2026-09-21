"""長期コホート・バックテストの計測指標。

`longterm_backtest._compute_metrics` をそのまま移設したもの。倍率系・勝率・
ベンチマーク比較の計算式は不変。損益依存の指標（Sharpe/Profit Factor/Calmar/DSR）
は `ClosedTrade.realized_pnl` から算出する（`metrics/stats.py` を参照）。
"""

from __future__ import annotations

import math

import pandas as pd

from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.portfolio import ClosedTrade
from src.backtest.metrics import stats
from src.backtest.metrics.overfitting import deflated_sharpe_ratio


def compute_longterm_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    config: LongtermBacktestConfig,
    benchmark: dict,
    closed: list[ClosedTrade],
) -> dict:
    """equity_df / trades_df / closed から計測指標を組み立てる。"""
    initial_cash = config.initial_cash
    start = config.start
    end = config.end

    if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
        final_value = float(equity_df["portfolio_value"].iloc[-1])
        equity_series = equity_df.set_index("date")["portfolio_value"]
        max_dd = stats.max_drawdown(equity_series)
    else:
        final_value = initial_cash
        max_dd = 0.0

    total_return = (final_value - initial_cash) / initial_cash if initial_cash > 0 else 0.0

    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1e-9)
    cagr = stats.cagr(initial_cash, final_value, years)

    n_trades = len(trades_df)
    if n_trades > 0:
        mm = trades_df["max_multiple"]
        n_2x = int((mm >= 2.0).sum())
        n_3x = int((mm >= 3.0).sum())
        n_5x = int((mm >= 5.0).sum())
        n_10x = int((mm >= 10.0).sum())

        realized = trades_df["multiple"]
        wins = trades_df[realized >= 1.0]
        losses = trades_df[realized < 1.0]
        win_rate = len(wins) / n_trades
        avg_win_multiple = float(wins["multiple"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["multiple"].mean() - 1.0) if not losses.empty else 0.0
        avg_held_days = float(trades_df["held_days"].mean())
    else:
        n_2x = n_3x = n_5x = n_10x = 0
        win_rate = avg_win_multiple = avg_loss = avg_held_days = 0.0

    bench_return = benchmark.get("total_return") if isinstance(benchmark, dict) else None
    alpha = (total_return - bench_return) if bench_return is not None else None

    pnls = [t.realized_pnl for t in closed]
    wins_pnl = [p for p in pnls if p >= 0]
    losses_pnl = [p for p in pnls if p < 0]
    spt = stats.sharpe_per_trade(pnls)
    trades_per_year = len(pnls) / years if pnls else 0.0
    pf = stats.profit_factor(wins_pnl, losses_pnl)

    result = {
        "initial_cash": round(initial_cash, 2),
        "final_cash": round(final_value, 2),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "max_drawdown": round(max_dd, 6),
        "num_trades": n_trades,
        "n_2x": n_2x,
        "n_3x": n_3x,
        "n_5x": n_5x,
        "n_10x": n_10x,
        "pct_2x": round(n_2x / n_trades, 4) if n_trades else 0.0,
        "pct_3x": round(n_3x / n_trades, 4) if n_trades else 0.0,
        "pct_5x": round(n_5x / n_trades, 4) if n_trades else 0.0,
        "pct_10x": round(n_10x / n_trades, 4) if n_trades else 0.0,
        "win_rate": round(win_rate, 4),
        "avg_win_multiple": round(avg_win_multiple, 4),
        "avg_loss": round(avg_loss, 6),
        "avg_held_days": round(avg_held_days, 1),
        "benchmark_ticker": benchmark.get("ticker") if isinstance(benchmark, dict) else None,
        "benchmark_return": bench_return,
        "alpha": round(alpha, 6) if alpha is not None else None,
        "sharpe_per_trade": round(spt, 6),
        "sharpe_ratio": round(stats.annualize_sharpe(spt, trades_per_year), 4),
        "profit_factor": round(pf, 4) if pf != math.inf else None,
        # max_dd はドローダウンが無いとき 0.0（負にならない）。その場合に
        # cagr/abs(max_dd) を計算すると 0 除算になるため、ドローダウン無し
        # （リスクが観測されていない）を意味する None を返す。
        "calmar_ratio": round(cagr / abs(max_dd), 4) if max_dd < 0 else None,
    }
    if config.n_trials > 0:
        result["dsr"] = deflated_sharpe_ratio(spt, config.n_trials, len(pnls))
    return result
