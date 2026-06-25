"""過学習ガード共通ヘルパー（DSR / PBO 警告閾値と PBO 算出）。

run_optimization / run_optuna_optimization の双方から参照される共有部品。
"""

import math

import numpy as np

from src.backtest.metrics import probability_of_backtest_overfitting
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Deflated Sharpe Ratio がこの値を下回る最良戦略は「多重比較バイアスを除くと
# 統計的に有意でない（過学習の疑い）」として警告する（ソフトゲート）。
_DSR_WARN_THRESHOLD = 0.95

# PBO (Probability of Backtest Overfitting) がこの値を超える最適化結果は
# 「インサンプル最良戦略が OOS で中央値以下に沈む確率が高い（過学習の疑い）」
# として警告する（ソフトゲート）。López de Prado (2014) の目安 0.5 を採用。
_PBO_WARN_THRESHOLD = 0.5


def _compute_pbo_from_fold_returns(
    fold_returns: list[list[float]],
    market: str,
    symbol: str,
) -> float:
    """候補ごとの fold 別リターン系列から PBO を計算しログ出力する。

    Walk-Forward の fold は時系列順なので、fold リターンを「期間リターン」と
    みなして CSCV を適用する（fold 粒度の近似。日次粒度はパイプラインが
    エクイティカーブを公開していないため将来課題）。

    Args:
        fold_returns: 候補（パラメータ組合せ / Optuna 試行）ごとの fold リターン系列
        market: ログ用マーケット識別子
        symbol: ログ用銘柄シンボル

    Returns:
        PBO（0〜1）。候補2未満・fold 不足のときは NaN。
    """
    series = [s for s in fold_returns if len(s) >= 2]
    if len(series) < 2:
        return float("nan")

    # 全候補を共通の最短 fold 数に揃える（エラー fold 等で長さが揃わない場合に備える）
    t = min(len(s) for s in series)
    matrix = np.array([s[:t] for s in series], dtype=float).T  # shape (T, N)

    # ブロック数は偶数かつ fold 数以下に制限（probability_of_backtest_overfitting
    # 側でも偶数化されるが、T を超えると NaN になるためここで上限を掛ける）
    n_splits = min(10, t - (t % 2))
    pbo = probability_of_backtest_overfitting(matrix, n_splits=n_splits)

    if math.isnan(pbo):
        return pbo
    logger.info(
        "[%s/%s] PBO=%.3f (candidates=%d, folds=%d, n_splits=%d)",
        market,
        symbol,
        pbo,
        matrix.shape[1],
        t,
        n_splits,
    )
    if pbo > _PBO_WARN_THRESHOLD:
        logger.warning(
            "[%s/%s] PBO=%.3f > %.2f: IS最良戦略が OOS で中央値以下に沈む確率が高い"
            "（過学習の疑い）",
            market,
            symbol,
            pbo,
            _PBO_WARN_THRESHOLD,
        )
    return pbo
