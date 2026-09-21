"""バックテスト指標の純粋統計関数。

単一銘柄バックテスト (`metrics/core.py`) と長期コホートバックテスト
(`longterm/`) の双方から呼べるよう、入力形状に依存しない統計計算のみを
切り出したモジュール。計算式は `core.py` からの移設時点のまま変更しない
（戦略ファクトリーの合否ゲートが sharpe_ratio の数値互換に依存するため）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sharpe_per_trade(pnl_list: list[float], risk_free_per_trade: float = 0.0) -> float:
    """取引単位の Sharpe（mean/std、年率化なし）を返す。

    DSR の入力（López de Prado の式は非年率の per-observation Sharpe を前提）と、
    年率化 Sharpe の素として使う。
    """
    if len(pnl_list) < 2:
        return 0.0
    arr = np.array(pnl_list, dtype=float)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float((arr.mean() - risk_free_per_trade) / std)


def annualize_sharpe(sharpe_per_trade: float, trades_per_year: float) -> float:
    """取引単位 Sharpe を実取引頻度で年率化する。"""
    if trades_per_year <= 0:
        return 0.0
    return float(sharpe_per_trade * math.sqrt(trades_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """資産曲線から最大ドローダウン（負の小数）を計算する"""
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def profit_factor(wins: list[float], losses: list[float]) -> float:
    """総利益 / 総損失（Profit Factor）を返す。

    `core.py` のインライン式と同じ集計だが、取引が皆無の場合のみ 0.0 を返す
    （`core.py` 側は `math.inf` を返し呼び出し側が None に変換するため、
    互換性維持の観点から `core.py` のインライン式は置き換えていない）。

    この差はそのまま出荷仕様になっている: 取引数ゼロのとき、
    `core.compute_metrics` は `profit_factor: None` を返すのに対し、
    `compute_longterm_metrics`（本関数を呼ぶ）は `profit_factor: 0.0` を返す。
    同じキー名で2つのバックテスターが異なる空状態の値を返す点に注意すること。
    """
    gross_profit = sum(w for w in wins if w > 0)
    gross_loss = abs(sum(loss for loss in losses if loss < 0))
    if gross_profit == 0 and gross_loss == 0:
        return 0.0
    return float(gross_profit / gross_loss) if gross_loss > 0 else math.inf


def cagr(initial: float, final: float, years: float) -> float:
    """年平均成長率（CAGR）を返す。

    `longterm/metrics.py` の式をそのまま関数化したもの。`years` は
    呼び出し側で `max(days / 365.25, 1e-9)` 等を計算して渡す。
    """
    return (final / initial) ** (1.0 / years) - 1.0 if final > 0 else -1.0
