"""
資本配分エンジン (R-301)

複数候補銘柄に対して保有比率を一括計算する。
信頼度・相関・レジーム・最大ポジション制約を統合し、
portfolio_backtest および order_execution_pipeline の双方で再利用できる設計にする。
"""

from __future__ import annotations

from typing import Optional

from config.settings import MAX_POSITION_RATE, MAX_POSITIONS, MAX_SECTOR_POSITIONS
from src.trading.types import AllocationCandidate, AllocationResult
from src.utils.logger import get_logger
from src.utils.sector_constraints import filter_by_sector_cap

logger = get_logger(__name__)

# レジーム別のポジションスケール係数
_REGIME_SCALE: dict[str, float] = {
    "bull": 1.0,
    "range": 0.6,
    "bear": 0.3,
}
_DEFAULT_REGIME_SCALE = 0.6


def _enc_corr_scale(n: int, avg_correlation: float) -> float:
    """ENC ベースの相関縮小係数を返す。

    ENC = n / (1 + avg_corr * (n-1)) で実効分散度を測定し、
    ENC/n (=0〜1) でポートフォリオ全体のウェイト予算を縮小する。
    完全無相関 (avg_corr=0) のとき scale=1.0、完全相関 (avg_corr=1) のとき scale=1/n。
    """
    if n <= 1 or avg_correlation <= 0.0:
        return 1.0
    enc = n / (1.0 + avg_correlation * (n - 1))
    return enc / n


def allocate_capital(
    candidates: list[AllocationCandidate],
    *,
    market_regime: Optional[str] = None,
    avg_correlation: float = 0.0,
    max_positions: int = MAX_POSITIONS,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
    max_weight_per_symbol: float = MAX_POSITION_RATE,
    min_diff_ratio: float = 0.0,
) -> list[AllocationResult]:
    """複数候補銘柄に対して推奨保有比率を一括計算する。

    Args:
        candidates: 予測スナップショットのリスト。
        market_regime: 市場レジーム ("bull" / "range" / "bear")。
            None の場合は "range" 相当スケールを適用する。
        avg_correlation: 候補銘柄間の平均絶対相関係数 [0, 1]。
            correlation_risk.compute_enc を用いて事前計算した値を渡す。
            高相関ほどポートフォリオ全体のウェイト総量を圧縮する。
        max_positions: 最大保有銘柄数上限。
        max_sector_positions: セクターごとの最大銘柄数。
        max_weight_per_symbol: 1銘柄あたりの最大ウェイト [0, 1]。
        min_diff_ratio: この値以下の diff_ratio を持つ候補を除外する。

    Returns:
        AllocationResult のリスト（weight の降順）。
        除外された銘柄は結果に含まれない。
    """
    if not candidates:
        return []

    regime_scale = _REGIME_SCALE.get(market_regime or "", _DEFAULT_REGIME_SCALE)
    logger.info(
        "[allocator] 配分計算開始: candidates=%d regime=%s regime_scale=%.2f avg_corr=%.3f",
        len(candidates),
        market_regime,
        regime_scale,
        avg_correlation,
    )

    # ステップ1: 最小リターン閾値でフィルタ
    filtered = [c for c in candidates if c.diff_ratio > min_diff_ratio]
    if not filtered:
        logger.info("[allocator] min_diff_ratio フィルタ後に候補なし")
        return []

    # ステップ2: 生スコア = diff_ratio * confidence_ratio でソート
    filtered.sort(key=lambda c: c.diff_ratio * c.confidence_ratio, reverse=True)

    # ステップ3: セクター上限でフィルタ
    filtered = filter_by_sector_cap(
        filtered,
        max_sector_positions=max_sector_positions,
        sector_getter=lambda c: c.sector,
    )

    # ステップ4: 最大ポジション数に絞る
    filtered = filtered[:max_positions]

    if not filtered:
        logger.info("[allocator] ポジション制約フィルタ後に候補なし")
        return []

    n = len(filtered)

    # ステップ5: ポートフォリオ予算の計算
    # ENC ベースの相関縮小: 高相関ほど総ウェイト上限を圧縮する
    corr_scale = _enc_corr_scale(n, avg_correlation)
    # レジーム縮小: 弱気相場ほど総ウェイトを圧縮する
    # 最大予算 = 銘柄数 × 1銘柄上限 × 相関縮小 × レジーム縮小、ただし 1.0 を超えない
    portfolio_budget = min(1.0, max_weight_per_symbol * n) * corr_scale * regime_scale

    # ステップ6: raw_score で比例配分して予算内に収める
    raw_scores = [max(0.0, c.diff_ratio * c.confidence_ratio) for c in filtered]
    total_score = sum(raw_scores)

    if total_score <= 0.0:
        logger.warning("[allocator] 合計スコアが 0 のため配分不可")
        return []

    # スコア比例で予算を割り当て、1銘柄上限でキャップ
    weights: list[float] = []
    for score in raw_scores:
        w = (score / total_score) * portfolio_budget
        w = min(w, max_weight_per_symbol)
        weights.append(w)

    results: list[AllocationResult] = []
    for candidate, raw_score, weight in zip(filtered, raw_scores, weights):
        results.append(
            AllocationResult(
                market=candidate.market,
                symbol=candidate.symbol,
                weight=weight,
                raw_score=raw_score,
                confidence_ratio=candidate.confidence_ratio,
                sector=candidate.sector,
                regime_scale=regime_scale,
            )
        )

    results.sort(key=lambda r: r.weight, reverse=True)

    logger.info(
        "[allocator] 配分完了: 銘柄数=%d 合計ウェイト=%.3f",
        len(results),
        sum(r.weight for r in results),
    )
    return results
