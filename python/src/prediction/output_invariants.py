"""予測出力の健全性チェック（出力 invariant）。

#612 / #613 / #615 に共通する「もっともらしい出力を出しながら中身が
縮退している」失敗モードを検知する。

DB もネットワークも触れない純関数として実装する。入力は「要求したモデル名」
「ロードできたモデル名」「今回の予測結果」「前回ラン統計」の4つのみ。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.prediction.types import PredictionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 閾値定数
# ---------------------------------------------------------------------------

# A-2: model_count が期待値未満の銘柄が占める割合。これ以上で違反。
DEGRADED_SYMBOL_RATIO_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRunStats:
    """1ラン分の集計値。前回ランとの比較に使う。"""

    symbol_count: int
    median_model_count: float
    diff_ratio_stdev: float


@dataclass(frozen=True)
class InvariantViolation:
    """不変条件の違反 1 件。"""

    violation_id: str
    description: str
    observed: float
    threshold: float


@dataclass
class InvariantReport:
    """1ラン分の評価結果。"""

    violations: list[InvariantViolation] = field(default_factory=list)
    stats: Optional[PredictionRunStats] = None
    compared_with_previous: bool = False

    @property
    def has_violation(self) -> bool:
        return bool(self.violations)

    @property
    def violation_ids(self) -> list[str]:
        return [v.violation_id for v in self.violations]

    def as_details(self) -> dict:
        """alert_service へ渡すプリミティブな内訳。

        alert_service は src.utils 層にあり src.prediction を import できない
        （.importlinter Contract 1）。そのため型ではなく dict で渡す。
        """
        return {
            "compared_with_previous": self.compared_with_previous,
            "symbol_count": self.stats.symbol_count if self.stats else 0,
            "median_model_count": self.stats.median_model_count if self.stats else 0.0,
            "diff_ratio_stdev": self.stats.diff_ratio_stdev if self.stats else 0.0,
            "violations": [
                {
                    "id": v.violation_id,
                    "description": v.description,
                    "observed": v.observed,
                    "threshold": v.threshold,
                }
                for v in self.violations
            ],
        }


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------


def build_run_stats(
    model_counts: Sequence[int], diff_ratios: Sequence[float]
) -> Optional[PredictionRunStats]:
    """ラン集計を作る。銘柄が 0 件なら None。

    標準偏差は母標準偏差（pstdev）を使う。標本標準偏差だと 1 銘柄で
    StatisticsError になるため。
    """
    if not model_counts:
        return None

    return PredictionRunStats(
        symbol_count=len(model_counts),
        median_model_count=float(statistics.median(model_counts)),
        diff_ratio_stdev=float(statistics.pstdev(diff_ratios)) if diff_ratios else 0.0,
    )


def build_run_stats_from_results(
    output_rows: Sequence[PredictionResult],
) -> Optional[PredictionRunStats]:
    """PredictionResult のリストからラン集計を作る。"""
    return build_run_stats(
        [r.model_count for r in output_rows],
        [r.diff_ratio for r in output_rows],
    )


# ---------------------------------------------------------------------------
# 絶対値チェック（前回ラン不要）
# ---------------------------------------------------------------------------


def _check_absolute(
    requested_model_names: Sequence[str],
    loaded_model_names: Sequence[str],
    output_rows: Sequence[PredictionResult],
) -> list[InvariantViolation]:
    violations: list[InvariantViolation] = []

    # A-1: モデルロード欠損
    # A-2 の期待値はロード成功数から導くため、ロード自体が減ると閾値も
    # 一緒に下がって縮退が無言化する。その穴を塞ぐのがこのルール。
    requested_count = len(requested_model_names)
    loaded_count = len(loaded_model_names)
    if loaded_count < requested_count:
        missing = sorted(set(requested_model_names) - set(loaded_model_names))
        violations.append(
            InvariantViolation(
                violation_id="A-1",
                description=f"モデルのロードに失敗: {', '.join(missing)}",
                observed=float(loaded_count),
                threshold=float(requested_count),
            )
        )

    # A-3: 予測 0 件
    # predict_all_unified の wrapper は例外を握って None を返すため、
    # 全銘柄が失敗しても空リストになるだけで例外にならない（#612 の経路）。
    if not output_rows:
        violations.append(
            InvariantViolation(
                violation_id="A-3",
                description="予測結果が 0 件",
                observed=0.0,
                threshold=1.0,
            )
        )
        return violations

    # A-2: アンサンブル縮退
    if loaded_count > 0:
        degraded = sum(1 for r in output_rows if r.model_count < loaded_count)
        ratio = degraded / len(output_rows)
        if ratio >= DEGRADED_SYMBOL_RATIO_THRESHOLD:
            violations.append(
                InvariantViolation(
                    violation_id="A-2",
                    description=(
                        f"アンサンブル縮退: {degraded}/{len(output_rows)} 銘柄が "
                        f"{loaded_count} モデル中 {loaded_count} 未満で予測されている"
                    ),
                    observed=ratio,
                    threshold=DEGRADED_SYMBOL_RATIO_THRESHOLD,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def evaluate_output_invariants(
    requested_model_names: Sequence[str],
    loaded_model_names: Sequence[str],
    output_rows: Sequence[PredictionResult],
    previous_stats: Optional[PredictionRunStats] = None,
) -> InvariantReport:
    """予測出力の不変条件を評価する。

    Args:
        requested_model_names: 要求したモデル名（A-1 の比較基準）
        loaded_model_names: ロードに成功したモデル名（A-2 の期待値）
        output_rows: 今回ランの予測結果
        previous_stats: 前回ラン統計。None なら急変チェックをスキップする

    Returns:
        InvariantReport
    """
    stats = build_run_stats_from_results(output_rows)
    violations = _check_absolute(requested_model_names, loaded_model_names, output_rows)

    return InvariantReport(
        violations=violations,
        stats=stats,
        compared_with_previous=False,
    )
