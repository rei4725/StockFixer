"""戦略ファクトリー: ポートフォリオ日次リターンに基づくゲート用の統計量。

ゲートの Sharpe は、有効銘柄を等金額で保有したポートフォリオの日次リターンから算出する。
取引リターンをプールして取引頻度で年率化する方式は取引を独立な賭けとみなすため、
市場ファクターのように分散で消えないリスクを無視し、保有の多い戦略を過大評価していた
（合成戦略による分類テスト tests/unit/backtest/test_factory_gate_classification.py で実証）。

チャンピオン比較は点推定の大小ではなく、同じ日付の日次リターン同士で Sharpe の差の
有意性を検定する（Jobson-Korkie 検定の Memmel 補正）。点推定の比較では、チャンピオンと
同等の候補が推定誤差だけで約半数合格していた。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    """正規化 equity の日次リターン（先頭の NaN は除く）。"""
    if equity is None or len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.pct_change().iloc[1:]


def per_period_sharpe(returns: pd.Series) -> float:
    """非年率の 1 期間あたり Sharpe。算出不能なら NaN。"""
    if returns is None or len(returns) < 2:
        return math.nan
    std = float(returns.std(ddof=1))
    if not std > 0:
        return math.nan
    return float(returns.mean()) / std


def annualized_sharpe(returns: pd.Series) -> float:
    """日次リターンの年率 Sharpe。算出不能なら NaN。"""
    sr = per_period_sharpe(returns)
    return sr * math.sqrt(TRADING_DAYS) if not math.isnan(sr) else math.nan


def sharpe_difference_z(candidate: pd.Series, champion: pd.Series) -> float:
    """候補とチャンピオンの Sharpe の差の z 値（Jobson-Korkie / Memmel 2003）。

    共通の日付に揃えた日次リターンで、SR_c - SR_ch の標準誤差を両者の相関込みで評価する:
        Var ≈ (1/T) * [2 - 2ρ + 0.5 (SR_c^2 + SR_ch^2 - 2 SR_c SR_ch ρ^2)]
    SR は非年率の 1 期間あたり値。正の大きな値ほど「候補がチャンピオンより有意に良い」。
    算出不能（共通日付が足りない・分散ゼロ）なら NaN。
    """
    if candidate is None or champion is None:
        return math.nan
    joined = pd.concat([candidate, champion], axis=1, join="inner").dropna()
    n_obs = len(joined)
    if n_obs < 3:
        return math.nan
    a = joined.iloc[:, 0]
    b = joined.iloc[:, 1]
    sr_a = per_period_sharpe(a)
    sr_b = per_period_sharpe(b)
    if math.isnan(sr_a) or math.isnan(sr_b):
        return math.nan
    rho = float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])
    var = (2 - 2 * rho + 0.5 * (sr_a**2 + sr_b**2 - 2 * sr_a * sr_b * rho**2)) / n_obs
    if not var > 0:
        return math.nan
    return (sr_a - sr_b) / math.sqrt(var)


def return_moments(returns: pd.Series) -> tuple[float, float]:
    """DSR に渡す歪度と尖度（正規分布なら 0 と 3）。算出不能なら正規分布の値。"""
    if returns is None or len(returns) < 4:
        return 0.0, 3.0
    values = returns.to_numpy(dtype=float)
    centered = values - values.mean()
    variance = float(np.mean(centered**2))
    if not variance > 0:
        return 0.0, 3.0
    # 母集団モーメント（標本数が数百あるためバイアス補正は不要）
    skew = float(np.mean(centered**3)) / variance**1.5
    kurt = float(np.mean(centered**4)) / variance**2
    return skew, kurt
