"""
戦略ファクトリー Phase 1（#369）: 合格ゲート判定

factory.py から切り出した純粋なゲート判定（#704 でファイル行数ゲートに抵触したため）。
factory_aggregation.py と同じく、factory.py の肥大化を抑えるための分離であり
ロジックは変更していない。

apply_gate は `src.backtest.factory` からも再エクスポートしているため、
既存の `from src.backtest.factory import apply_gate` は引き続き動作する。
"""

from __future__ import annotations

import math

from config.settings import (
    FACTORY_GATE_CHAMPION_MARGIN,
    FACTORY_GATE_MAX_DRAWDOWN,
    FACTORY_GATE_MIN_DSR,
    FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS,
    FACTORY_GATE_MIN_TRADES,
)
from src.backtest.types import FactoryEvaluation


def apply_gate(evaluation: FactoryEvaluation, champion_sharpe: float) -> None:
    """ゲート条件を判定し evaluation.gate_passed / gate_reasons を更新する。

    PBO はバッチ全体で1値となる性質上、per-hypothesis ゲートに使うと「一晩全滅」に
    なるため、ここでは判定しない（バッチ診断としてレポート/通知に警告表示する）。
    DSR はトレード単位 Sharpe（年率化を打ち消した値）で算出済みのため飽和しない。
    有効銘柄数（銘柄あたり最低取引数を満たした銘柄の数）が下限未満の場合も不合格とする。
    合計取引数だけでは「2銘柄 × 20取引」のような極端な集中を弾けないため（#625）。

    champion_sharpe が NaN の場合は「対照群が全滅してチャンピオン比較ができない」ことを
    意味する。以前はこの条件を丸ごとスキップしていた（fail-open）が、最も強いゲートが
    無言で外れて質の悪い仮説が通ってしまうため、fail-closed（不合格）に倒す（#627）。
    """
    reasons: list[str] = []
    if evaluation.num_trades < FACTORY_GATE_MIN_TRADES:
        reasons.append(f"num_trades {evaluation.num_trades} < {FACTORY_GATE_MIN_TRADES}")
    if evaluation.n_effective_symbols < FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS:
        reasons.append(
            f"effective_symbols {evaluation.n_effective_symbols}"
            f" < {FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS}"
        )
    if math.isnan(evaluation.dsr) or evaluation.dsr < FACTORY_GATE_MIN_DSR:
        reasons.append(f"dsr {evaluation.dsr:.3f} < {FACTORY_GATE_MIN_DSR}")
    _append_drawdown_reason(evaluation, reasons)
    if math.isnan(champion_sharpe):
        reasons.append("champion_sharpe が NaN（対照群が全滅しチャンピオン比較不能）のため不合格")
    else:
        required = champion_sharpe * FACTORY_GATE_CHAMPION_MARGIN
        if evaluation.sharpe_ratio <= required:
            reasons.append(
                f"sharpe {evaluation.sharpe_ratio:.3f} <= champion×{FACTORY_GATE_CHAMPION_MARGIN}"
                f" ({required:.3f})"
            )
    evaluation.gate_reasons = reasons
    evaluation.gate_passed = not reasons


def _append_drawdown_reason(evaluation: FactoryEvaluation, reasons: list[str]) -> None:
    """DD ゲートはポートフォリオDDで判定する。

    最悪銘柄DD（min over symbols）は有効銘柄数を増やすほど必ず悪化する最小値統計で
    あり、等金額で20銘柄以上を保有する運用者が経験する数字ではない。閾値
    FACTORY_GATE_MAX_DRAWDOWN は据え置き、比較する「量」だけを是正する。

    equity 曲線が1本も得られずポートフォリオDDが算出できなかった場合（NaN）は、
    ゲートを無言で外さないよう従来の最悪銘柄DDにフォールバックする。
    """
    value = evaluation.portfolio_max_drawdown
    label = "portfolio_max_drawdown"
    if math.isnan(value):
        value = evaluation.max_drawdown
        label = "max_drawdown"
    if value < FACTORY_GATE_MAX_DRAWDOWN:
        reasons.append(f"{label} {value:.3f} < {FACTORY_GATE_MAX_DRAWDOWN}")
