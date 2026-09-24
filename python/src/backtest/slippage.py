"""
動的スリッページモデル（R-210）

平方根市場インパクトモデルで注文サイズ・出来高連動のスリッページを推定する。

市場インパクト係数 alpha の実績校正は持たない（#740）。paper_real_diff には
発注株数も平均出来高も記録されておらず、slippage = alpha * sqrt(qty / ADV) の
alpha を逆算できないため。校正が必要になったら、まず qty と ADV を記録すること。

使い方:
    # 発注時のスリッページ推定
    slip = estimate_slippage(order_qty=200, price=1500.0, avg_daily_volume=50_000, alpha=0.10)
    cost_pct = slip  # スリッページ率（例: 0.003 = 0.3%）

    # バックテスト用の slippage_fn を生成
    fn = make_slippage_fn(alpha=0.10)
    backtester = Backtester(..., slippage_fn=fn)
"""

from __future__ import annotations

import math
from typing import Callable

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_ALPHA = 0.10


def estimate_slippage(
    order_qty: int,
    price: float,
    avg_daily_volume: float,
    alpha: float = _DEFAULT_ALPHA,
) -> float:
    """
    平方根市場インパクトモデルによるスリッページ率推定。

    slippage = alpha * sqrt(order_qty / avg_daily_volume)

    参加率が高い（大口注文/低流動性）ほどスリッページが増加する。
    参加率が 0 に近い場合はほぼゼロになる。

    Args:
        order_qty: 発注株数
        price: 発注価格（将来の出来高金額連動に備えて受け取る）
        avg_daily_volume: 過去 N 日の平均出来高（株数）
        alpha: 市場インパクト係数（デフォルト 0.10）

    Returns:
        スリッページ率（0.0〜）
    """
    if avg_daily_volume <= 0 or order_qty <= 0 or price <= 0:
        return 0.0
    participation_rate = order_qty / avg_daily_volume
    return float(alpha * math.sqrt(participation_rate))


def make_slippage_fn(alpha: float = _DEFAULT_ALPHA) -> Callable[[int, float, float], float]:
    """
    バックテスター用スリッページ関数ファクトリー。

    Returns:
        (order_qty, price, avg_daily_volume) -> slippage_rate を受け取る callable
    """

    def _fn(order_qty: int, price: float, avg_daily_volume: float) -> float:
        return estimate_slippage(order_qty, price, avg_daily_volume, alpha=alpha)

    return _fn
