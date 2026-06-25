"""バックテスト過学習・統計系メトリクス

Deflated Sharpe Ratio / PBO（CSCV）/ モンテカルロ equity シミュレーション。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm as _norm


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (López de Prado, 2018).

    Args:
        sharpe:    算出済みの Sharpe Ratio
        n_trials:  グリッドサーチの試行数（例: パラメータ組み合わせ数）
        n_obs:     バックテストの取引回数（observations）
        skewness:  取引リターンの歪度（デフォルト 0 = 正規分布近似）
        kurtosis:  取引リターンの尖度（デフォルト 3 = 正規分布近似）

    Returns:
        0〜1 の確率値。1 に近いほど「偶然ではないスキルベースの成績」。
    """
    EULER_GAMMA = 0.5772156649

    var = (1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe**2) / max(n_obs - 1, 1)
    sr_std = math.sqrt(max(var, 0.0))
    if sr_std <= 0 or n_trials <= 0:
        return 0.0

    # n_trials=1 は多重比較なし: 期待最大値=0 (1試行のN(0,1)の期待値)
    if n_trials == 1:
        expected_max = 0.0
    else:
        expected_max = sr_std * (
            (1 - EULER_GAMMA) * _norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * _norm.ppf(1 - 1 / (n_trials * math.e))
        )
    z = (sharpe - expected_max) / sr_std
    return float(_norm.cdf(z))


def _sharpe_by_column(block: np.ndarray) -> np.ndarray:
    """各列（=各戦略候補）の Sharpe（mean/std）を返す。std=0 の列は 0。"""
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=1) if block.shape[0] > 1 else np.ones(block.shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, mean / std, 0.0)
    return np.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 10,
) -> float:
    """CSCV (Combinatorially Symmetric Cross-Validation) による PBO を計算する。

    López de Prado (2014) の手法。N 個の候補戦略（パラメータ組合せ等）の期間
    リターン行列から「インサンプル最良の戦略がアウトオブサンプルで中央値を下回る確率」
    を推定する。PBO が高い（>0.5 目安）ほど、最適化結果が過学習である可能性が高い。

    Args:
        returns_matrix: shape (T, N) の配列。T=期間数、N=候補戦略数。各列が1戦略の
            期間リターン系列（例: トレードごと or 日次リターン）。
        n_splits: 期間 T を分割するブロック数 S（偶数）。C(S, S/2) 通りの IS/OOS 分割を作る。

    Returns:
        PBO（0〜1）。候補が2未満・データ不足のときは NaN。
    """
    from itertools import combinations

    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return float("nan")
    if n_splits % 2 != 0:
        n_splits -= 1
    if n_splits < 2:
        return float("nan")

    T, N = M.shape
    block_size = T // n_splits
    if block_size < 1:
        return float("nan")

    # 末尾の端数は捨てて等分割
    blocks = [M[i * block_size : (i + 1) * block_size] for i in range(n_splits)]
    indices = list(range(n_splits))

    lambdas: list[float] = []
    for is_sel in combinations(indices, n_splits // 2):
        is_set = set(is_sel)
        oos_sel = [i for i in indices if i not in is_set]
        is_mat = np.vstack([blocks[i] for i in is_sel])
        oos_mat = np.vstack([blocks[i] for i in oos_sel])

        is_perf = _sharpe_by_column(is_mat)
        oos_perf = _sharpe_by_column(oos_mat)

        # IS 最良戦略
        n_star = int(np.argmax(is_perf))
        # その戦略の OOS 相対順位 ω = rank / (N+1)（rank: 1=最下位 … N=最上位）
        order = np.argsort(oos_perf, kind="stable")  # 昇順
        rank = int(np.where(order == n_star)[0][0]) + 1
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        lambdas.append(math.log(omega / (1.0 - omega)))

    if not lambdas:
        return float("nan")
    arr = np.asarray(lambdas)
    # λ <= 0 = IS最良が OOS で中央値以下 = 過学習の兆候
    return float(np.mean(arr <= 0.0))


def monte_carlo_equity(
    trade_pnl_list: list[float],
    initial_cash: float,
    n_simulations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    取引損益系列をランダムシャッフルして equity curve を n_simulations 回シミュレートし、
    最大ドローダウンと最終資産の分布統計を返す。

    Args:
        trade_pnl_list: 取引ごとの損益リスト（正=利益, 負=損失）
        initial_cash:   初期資金
        n_simulations:  シミュレーション回数
        confidence:     信頼水準（デフォルト 0.95 = 95%）
        seed:           乱数シード（再現性確保用）

    Returns:
        {
            "max_drawdown_mean":  最大ドローダウンの平均（負の小数）,
            "max_drawdown_p95":   最大ドローダウンの 95 パーセンタイル（負の小数）,
            "final_cash_p05":     最終資産の 5 パーセンタイル,
            "final_cash_p50":     最終資産の中央値,
            "final_cash_p95":     最終資産の 95 パーセンタイル,
        }
    """
    if not trade_pnl_list:
        return {
            "max_drawdown_mean": 0.0,
            "max_drawdown_p95": 0.0,
            "final_cash_p05": 0.0,
            "final_cash_p50": 0.0,
            "final_cash_p95": 0.0,
        }

    rng = np.random.default_rng(seed)
    arr = np.array(trade_pnl_list, dtype=float)

    max_dds: list[float] = []
    final_cashes: list[float] = []
    for _ in range(n_simulations):
        # 復元抽出（ブートストラップ）で equity curve を生成
        shuffled = rng.choice(arr, size=len(arr), replace=True)
        equity = np.empty(len(shuffled) + 1)
        equity[0] = initial_cash
        equity[1:] = initial_cash + np.cumsum(shuffled)

        roll_max = np.maximum.accumulate(equity)
        dd = (equity - roll_max) / np.where(roll_max > 0, roll_max, 1.0)
        max_dds.append(float(dd.min()))
        final_cashes.append(float(equity[-1]))

    pct = int(confidence * 100)
    return {
        "max_drawdown_mean": float(np.mean(max_dds)),
        "max_drawdown_p95": float(np.percentile(max_dds, pct)),
        "final_cash_p05": float(np.percentile(final_cashes, 5)),
        "final_cash_p50": float(np.percentile(final_cashes, 50)),
        "final_cash_p95": float(np.percentile(final_cashes, pct)),
    }
