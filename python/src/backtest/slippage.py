"""
動的スリッページモデル（R-210）

平方根市場インパクトモデルで注文サイズ・出来高連動のスリッページを推定する。
paper_real_diff テーブルの実績データから市場インパクト係数 alpha を校正できる。

使い方:
    # alpha をデータから校正
    with _db_connection() as con:
        alpha = calibrate_alpha(con)

    # 発注時のスリッページ推定
    slip = estimate_slippage(order_qty=200, price=1500.0, avg_daily_volume=50_000, alpha=alpha)
    cost_pct = slip  # スリッページ率（例: 0.003 = 0.3%）

    # バックテスト用の slippage_fn を生成
    fn = make_slippage_fn(alpha=alpha)
    backtester = Backtester(..., slippage_fn=fn)
"""

from __future__ import annotations

import math
from typing import Callable

import duckdb

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_ALPHA = 0.10
_ALPHA_CLIP_MIN = 0.0
_ALPHA_CLIP_MAX = 0.05


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


def calibrate_alpha(con: duckdb.DuckDBPyConnection, min_samples: int = 10) -> float:
    """
    paper_real_diff テーブルの実績スリッページから alpha を推定する。

    約定価格と予測シグナル価格の乖離の中央値を参照し、
    ゼロ以下・外れ値は除外した上で [0%, 5%] にクリップする。

    Args:
        con: DuckDB 接続
        min_samples: 推定に必要な最低サンプル数（不足時はデフォルト値を返す）

    Returns:
        推定 alpha 値
    """
    try:
        rows = con.execute("""
            SELECT signal_price, actual_price, side
            FROM paper_real_diff
            WHERE actual_price IS NOT NULL
              AND signal_price  IS NOT NULL
              AND signal_price  > 0
              AND actual_price  > 0
            """).fetchall()
    except Exception as e:
        logger.warning("paper_real_diff 読み込み失敗、デフォルト alpha を使用: %s", e)
        return _DEFAULT_ALPHA

    if len(rows) < min_samples:
        logger.debug(
            "paper_real_diff サンプル数不足 (%d < %d)、デフォルト alpha=%s を使用",
            len(rows),
            min_samples,
            _DEFAULT_ALPHA,
        )
        return _DEFAULT_ALPHA

    slippages = []
    for signal_price, actual_price, side in rows:
        sp, ap = float(signal_price), float(actual_price)
        # BUY(side=1): actual > signal なら不利（コストが高い）
        # SELL(side=2) / SHORT_COVER(side=4): actual < signal なら不利
        if side == 1:
            slip = (ap - sp) / sp
        else:
            slip = (sp - ap) / sp
        if slip >= 0:
            slippages.append(slip)

    if not slippages:
        return _DEFAULT_ALPHA

    slippages.sort()
    median_slip = slippages[len(slippages) // 2]
    alpha = float(min(_ALPHA_CLIP_MAX, max(_ALPHA_CLIP_MIN, median_slip)))
    logger.info(
        "alpha 校正完了: サンプル=%d 中央値スリッページ=%.4f alpha=%.4f",
        len(slippages),
        median_slip,
        alpha,
    )
    return alpha


def make_slippage_fn(alpha: float = _DEFAULT_ALPHA) -> Callable[[int, float, float], float]:
    """
    バックテスター用スリッページ関数ファクトリー。

    Returns:
        (order_qty, price, avg_daily_volume) -> slippage_rate を受け取る callable
    """

    def _fn(order_qty: int, price: float, avg_daily_volume: float) -> float:
        return estimate_slippage(order_qty, price, avg_daily_volume, alpha=alpha)

    return _fn


def get_calibrated_slippage_fn(
    con: duckdb.DuckDBPyConnection,
) -> Callable[[int, float, float], float]:
    """
    paper_real_diff から alpha を校正したスリッページ関数を返す。

    Args:
        con: DuckDB 接続

    Returns:
        校正済みスリッページ関数
    """
    alpha = calibrate_alpha(con)
    return make_slippage_fn(alpha=alpha)
