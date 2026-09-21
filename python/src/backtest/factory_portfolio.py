"""戦略ファクトリー: 有効銘柄を等金額で保有したときのポートフォリオ equity 合成。

ゲートの DD 指標として最悪銘柄DD（`min` over symbols）を使うと、有効銘柄数を
増やすほど必ず悪化する最小値統計を閾値と比べることになり、等金額で多銘柄を
保有する運用者が実際に経験する数字から乖離する。ここでは銘柄別の日次
mark-to-market equity を等金額で合成し、その曲線の DD を算出する。

銘柄別シミュレーションはそれぞれ initial_cash から独立に開始するため、
各曲線を initial_cash で正規化してから平均する。DD はスケール不変なので
平均でも合計でも同値である。
"""

from __future__ import annotations

import math
from typing import Sequence

import pandas as pd

from src.backtest.metrics import _max_drawdown

# まだ記録が始まっていない日付は「建玉を持たず現金のまま」とみなす正規化 equity。
_CASH_LEVEL = 1.0


def build_portfolio_equity(curves: Sequence[pd.Series], initial_cash: float) -> pd.Series:
    """銘柄別の日次 equity 曲線を等金額で合成した正規化 equity を返す。

    Args:
        curves: 銘柄別の日次 mark-to-market equity（金額建て）。空 Series は無視する。
        initial_cash: 各銘柄シミュレーションの初期資金。

    Returns:
        全銘柄の日付和集合をインデックスとする正規化 equity（初期値 1.0 基準）。
        使える曲線が1本も無い場合は空 Series。
    """
    usable = [c for c in curves if c is not None and not c.empty]
    if not usable or initial_cash <= 0:
        return pd.Series(dtype=float)

    normalized = [c.astype(float) / initial_cash for c in usable]
    index = normalized[0].index
    for curve in normalized[1:]:
        index = index.union(curve.index)
    index = index.sort_values()

    # 記録開始前は現金（1.0）、記録の隙間・終了後は直前値で前方補完する。
    aligned = [c.reindex(index).ffill().fillna(_CASH_LEVEL) for c in normalized]
    return pd.concat(aligned, axis=1).mean(axis=1)


def portfolio_max_drawdown(curves: Sequence[pd.Series], initial_cash: float) -> float:
    """等金額ポートフォリオの最大ドローダウン（負の小数）を返す。

    使える曲線が1本も無い場合は NaN を返す。呼び出し側はこれを検出して
    従来の最悪銘柄DDへフォールバックする。
    """
    equity = build_portfolio_equity(curves, initial_cash)
    if equity.empty:
        return math.nan
    return _max_drawdown(equity)
