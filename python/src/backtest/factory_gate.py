"""
戦略ファクトリー Phase 1（#369）: 合格ゲート判定

factory.py から切り出した純粋なゲート判定（#704 でファイル行数ゲートに抵触したため）。

ゲートの論理的な正しさは、正解ラベル付きの合成戦略で統計的に検証している
（tests/unit/backtest/test_factory_gate_classification.py）。判定の「量」を変える場合は
このテストが緑のままであることを確認すること。

apply_gate は `src.backtest.factory` からも再エクスポートしているため、
既存の `from src.backtest.factory import apply_gate` は引き続き動作する。
"""

from __future__ import annotations

import math
from typing import Optional

from config.settings import (
    FACTORY_GATE_CHAMPION_MIN_Z,
    FACTORY_GATE_MAX_DRAWDOWN,
    FACTORY_GATE_MIN_DSR,
    FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS,
    FACTORY_GATE_MIN_TRADES,
)
from src.backtest.factory_significance import per_period_sharpe, return_moments, sharpe_difference_z
from src.backtest.metrics import deflated_sharpe_ratio
from src.backtest.types import FactoryEvaluation


def portfolio_dsr(evaluation: FactoryEvaluation, n_trials: int) -> float:
    """ポートフォリオ日次リターンの Deflated Sharpe Ratio。算出不能なら NaN。

    観測数は日数、歪度・尖度は実測値を使う。以前は 1 取引あたり Sharpe と取引数で
    算出していたが、同じ日に多数の銘柄で建つ取引は独立な観測ではないため、取引の多い
    戦略ほど DSR が 1.0 に張り付き、多重比較の補正として機能していなかった。
    """
    returns = evaluation.portfolio_returns
    if returns is None:
        return math.nan
    sr = per_period_sharpe(returns)
    if math.isnan(sr):
        return math.nan
    skew, kurt = return_moments(returns)
    return deflated_sharpe_ratio(sr, max(n_trials, 1), len(returns), skew, kurt)


def apply_gate(evaluation: FactoryEvaluation, champion: Optional[FactoryEvaluation]) -> None:
    """ゲート条件を判定し evaluation.gate_passed / gate_reasons / champion_z を更新する。

    evaluation.dsr は呼び出し側が portfolio_dsr で設定しておくこと。

    PBO はバッチ全体で1値となる性質上、per-hypothesis ゲートに使うと「一晩全滅」に
    なるため、ここでは判定しない（バッチ診断としてレポート/通知に警告表示する）。
    有効銘柄数（銘柄あたり最低取引数を満たした銘柄の数）が下限未満の場合も不合格とする。
    合計取引数だけでは「2銘柄 × 20取引」のような極端な集中を弾けないため（#625）。

    champion が None の場合は「対照群が全滅してチャンピオン比較ができない」ことを
    意味する。最も強いゲートが無言で外れないよう fail-closed（不合格）に倒す（#627）。
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
    if champion is None:
        reasons.append("チャンピオンが無い（対照群が全滅しチャンピオン比較不能）ため不合格")
    else:
        _append_champion_reason(evaluation, champion, reasons)
    evaluation.gate_reasons = reasons
    evaluation.gate_passed = not reasons


def _append_champion_reason(
    evaluation: FactoryEvaluation, champion: FactoryEvaluation, reasons: list[str]
) -> None:
    """チャンピオン比較は、同じ日付の日次リターン同士の Sharpe 差の検定で行う。

    点推定の大小比較（候補 Sharpe > チャンピオン Sharpe）では、推定誤差だけで
    チャンピオンと同等の候補が約半数合格していた。z 値が FACTORY_GATE_CHAMPION_MIN_Z
    を超えた場合のみ「チャンピオンより有意に良い」とみなす。
    日次リターンが無く検定できない場合は不合格に倒す。
    """
    z = sharpe_difference_z(evaluation.portfolio_returns, champion.portfolio_returns)
    evaluation.champion_z = z
    if math.isnan(z):
        reasons.append("チャンピオンとの Sharpe 差を検定できない（日次リターンが無い）")
    elif z <= FACTORY_GATE_CHAMPION_MIN_Z:
        reasons.append(
            f"portfolio_sharpe {evaluation.portfolio_sharpe_ratio:.3f} vs champion "
            f"{champion.portfolio_sharpe_ratio:.3f}: 差の z={z:.2f} <= {FACTORY_GATE_CHAMPION_MIN_Z}"
        )


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
