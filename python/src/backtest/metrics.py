"""
バックテスト評価指標

equity_curve (日次・取引ごとの資産曲線 pd.Series) を入力として
各種パフォーマンス指標を計算する関数群。
"""

from __future__ import annotations

import math
import pandas as pd
import numpy as np


def compute_metrics(
    trade_log: pd.DataFrame,
    initial_cash: float,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> dict:
    """
    取引ログから主要メトリクスを一括計算する。

    Args:
        trade_log: Backtester.simulate_trading が返す DataFrame
                   (columns: date, action, price, qty, cash)
        initial_cash: 初期資金
        risk_free_rate: 無リスク金利（年率、小数）
        trading_days_per_year: 1年の取引日数（Sharpe計算用）

    Returns:
        dict: 各指標の辞書
    """
    if trade_log is None or trade_log.empty:
        return _empty_metrics(initial_cash)

    # --- equity curve（決済後のキャッシュ推移）---
    # buy直後はキャッシュが激減するため sell/final_sell 時点のキャッシュのみを使う
    if "action" in trade_log.columns:
        sell_log = trade_log[trade_log["action"].isin(["sell", "final_sell"])]
    else:
        sell_log = trade_log

    if sell_log.empty or "date" not in sell_log.columns:
        equity = pd.Series([initial_cash])
    else:
        equity = pd.concat([
            pd.Series([initial_cash], index=[sell_log.iloc[0]["date"]]),
            sell_log.set_index("date")["cash"],
        ])

    final_cash = equity.iloc[-1]
    total_return = (final_cash - initial_cash) / initial_cash

    # 取引ペア（buy/sell）を抽出して勝率・Profit Factor を計算
    wins, losses = _extract_trade_pnl(trade_log)
    num_trades = len(wins) + len(losses)
    win_rate = len(wins) / num_trades if num_trades > 0 else 0.0

    gross_profit = sum(w for w in wins if w > 0)
    gross_loss = abs(sum(l for l in losses if l < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf

    # --- Sharpe ratio（取引ごとのリターン列から） ---
    sharpe = _sharpe_ratio(wins + losses, risk_free_rate, trading_days_per_year)

    # --- Maximum Drawdown ---
    max_dd = _max_drawdown(equity)

    return {
        "final_cash": round(final_cash, 2),
        "total_return": round(total_return, 6),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != math.inf else None,
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
    }


# --- internal helpers ---

def _empty_metrics(initial_cash: float) -> dict:
    return {
        "final_cash": initial_cash,
        "total_return": 0.0,
        "num_trades": 0,
        "win_rate": 0.0,
        "profit_factor": None,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
    }


def _extract_trade_pnl(trade_log: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    buy → sell のペアから各トレードの損益を抽出する。

    Returns:
        (wins, losses): 正の損益リスト、負の損益リスト
    """
    buys: list[dict] = []
    wins: list[float] = []
    losses: list[float] = []

    for _, row in trade_log.iterrows():
        action = row.get("action", "")
        if action == "buy":
            buys.append({"price": row["price"], "qty": row["qty"]})
        elif action in ("sell", "final_sell") and buys:
            buy = buys.pop(0)
            pnl = (row["price"] - buy["price"]) * buy["qty"]
            (wins if pnl >= 0 else losses).append(pnl)

    return wins, losses


def _sharpe_ratio(
    pnl_list: list[float],
    risk_free_rate: float,
    trading_days_per_year: int,
) -> float:
    """取引単位の損益リストから Sharpe ratio を計算する"""
    if len(pnl_list) < 2:
        return 0.0
    arr = np.array(pnl_list)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    # 年率換算（取引回数ベース）
    trades_per_year = trading_days_per_year  # 近似値
    daily_rf = risk_free_rate / trades_per_year
    return float((mean - daily_rf) / std * math.sqrt(trades_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    """資産曲線から最大ドローダウン（負の小数）を計算する"""
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())
