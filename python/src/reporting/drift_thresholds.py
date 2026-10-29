"""ドリフト判定（閾値超過銘柄の選び出し）の共通部品。

日次のドリフト検査（orchestration/jobs/drift_check.py）と月次レポートの
閾値超過銘柄数（reporting/kpi.py）が、同じ設定値・同じ式で数えるために使う（#1089）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

# MAE の実効の線 = max(DRIFT_MAE_THRESHOLD, 全銘柄の MAE の中央値 × この倍率)
DRIFT_MAE_RELATIVE_RATIO_DEFAULT = "1.5"
# Hit Rate の実効の線 = min(DRIFT_HIT_RATE_THRESHOLD, 全銘柄の Hit Rate の中央値 - この幅)
# 20件の Hit Rate の標本のぶれ（p=0.45 で標準偏差 約0.11）の約2倍を幅にとる（#1045）
DRIFT_HIT_RATE_RELATIVE_MARGIN_DEFAULT = "0.20"
# 評価した銘柄のうち閾値超えの割合がこの線以上の日を NF-303-3 の Warn と数える（#1369）。
# 10-26〜10-29 のふだんの日は 23〜47 / 約700 銘柄（3〜7%）、10-25 より前の傷んだ日は
# 約694 / 700 銘柄（99%）。ふだんの日の上の端の約2倍にとる
DRIFT_WARN_SHARE_DEFAULT = "0.15"


@dataclass(frozen=True)
class DriftThresholds:
    """ドリフト判定の設定値。既定値は環境変数・system_config が無いときの値と同じ。"""

    mae_threshold: float = 0.02
    hit_rate_threshold: float = 0.45
    min_samples: int = 10
    mae_relative_ratio: float = float(DRIFT_MAE_RELATIVE_RATIO_DEFAULT)
    hit_rate_relative_margin: float = float(DRIFT_HIT_RATE_RELATIVE_MARGIN_DEFAULT)
    warn_share: float = float(DRIFT_WARN_SHARE_DEFAULT)


@dataclass(frozen=True)
class DriftSelection:
    """判定の結果。triggered は閾値超過の銘柄の行、線は中央値に合わせた後の実効の値。

    n_evaluated はサンプル数の下限を満たして判定した銘柄の数（Warn の割合の分母、#1369）。
    """

    triggered: pd.DataFrame
    mae_threshold: float
    hit_rate_threshold: float
    n_evaluated: int = 0

    @property
    def triggered_share(self) -> float:
        """判定した銘柄のうち閾値超過の割合。判定した銘柄が無ければ 0。"""
        return len(self.triggered) / self.n_evaluated if self.n_evaluated else 0.0


def load_drift_thresholds() -> DriftThresholds:
    """system_config（drift.*）を優先し、無ければ環境変数、さらに無ければ既定値で読む。"""
    from src.utils.db.system_config import get_config_value

    def _get(key: str, env: str, default: str) -> str:
        return str(get_config_value(f"drift.{key}", os.environ.get(env, default)))

    return DriftThresholds(
        mae_threshold=float(_get("mae_threshold", "DRIFT_MAE_THRESHOLD", "0.02")),
        hit_rate_threshold=float(_get("hit_rate_threshold", "DRIFT_HIT_RATE_THRESHOLD", "0.45")),
        min_samples=int(_get("min_samples", "DRIFT_MIN_SAMPLES", "10")),
        mae_relative_ratio=float(
            _get("mae_relative_ratio", "DRIFT_MAE_RELATIVE_RATIO", DRIFT_MAE_RELATIVE_RATIO_DEFAULT)
        ),
        hit_rate_relative_margin=float(
            _get(
                "hit_rate_relative_margin",
                "DRIFT_HIT_RATE_RELATIVE_MARGIN",
                DRIFT_HIT_RATE_RELATIVE_MARGIN_DEFAULT,
            )
        ),
        warn_share=float(_get("warn_share", "DRIFT_WARN_SHARE", DRIFT_WARN_SHARE_DEFAULT)),
    )


def select_drifted_symbols(summary: pd.DataFrame, thresholds: DriftThresholds) -> DriftSelection:
    """drift_summary の表から、サンプル数の下限を満たし実効の線を超えた銘柄を選ぶ。"""
    eligible = summary[summary["n_samples"] >= thresholds.min_samples]
    mae_threshold = thresholds.mae_threshold
    hit_rate_threshold = thresholds.hit_rate_threshold
    if not eligible.empty:
        # 全銘柄の MAE が同じ水準で高いときに全件が超えないよう、線を全銘柄の中央値の倍率まで上げる（#1020）
        mae_threshold = max(
            mae_threshold,
            float(eligible["mean_abs_error"].median()) * thresholds.mae_relative_ratio,
        )
        # 全銘柄の Hit Rate が同じ水準で低いときに半数が超えないよう、線を中央値から幅を引いた値まで下げる（#1045）
        hit_rate_threshold = min(
            hit_rate_threshold,
            float(eligible["direction_accuracy"].median()) - thresholds.hit_rate_relative_margin,
        )
    triggered = eligible[
        (eligible["mean_abs_error"] >= mae_threshold)
        | (eligible["direction_accuracy"] <= hit_rate_threshold)
    ]
    return DriftSelection(
        triggered=triggered,
        mae_threshold=mae_threshold,
        hit_rate_threshold=hit_rate_threshold,
        n_evaluated=len(eligible),
    )
