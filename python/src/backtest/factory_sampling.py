"""戦略ファクトリー（#369）: 探索空間の定義と仮説サンプラー

factory.py が行数ゲート（1ファイル600行）に抵触したため切り出した
（factory_aggregation.py / factory_gate.py と同じ分割方針）。
探索空間（ルール種別とパラメータグリッド）とサンプリングだけを持ち、
評価・ゲート判定には関与しない。

_RULE_CLASSES / _PARAM_GRID / sample_hypotheses / control_hypotheses は
`src.backtest.factory` からも再エクスポートしているため、既存の
`from src.backtest.factory import sample_hypotheses` は引き続き動作する。
"""

from __future__ import annotations

import random
from typing import Any, Optional

from src.backtest.rules import (
    BollingerBandRule,
    EMAMomentumRule,
    MACDRSIRule,
    RSIContrarianRule,
    VolatilityBreakoutRule,
    VolumeBreakoutRule,
)
from src.backtest.types import FactoryHypothesis
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 探索空間
# ---------------------------------------------------------------------------

_RULE_CLASSES: dict[str, type] = {
    "volume_breakout": VolumeBreakoutRule,
    "ema_momentum": EMAMomentumRule,
    "rsi_contrarian": RSIContrarianRule,
    "bollinger_band": BollingerBandRule,
    "macd_rsi": MACDRSIRule,
    "volatility_breakout": VolatilityBreakoutRule,
}

# 各ルールのパラメータ候補（先頭がデフォルト相当 = 対照に使用）
_PARAM_GRID: dict[str, list[dict[str, Any]]] = {
    "volume_breakout": [
        {"volume_ratio": 2.0, "breakout_window": 20},
        {"volume_ratio": 1.5, "breakout_window": 20},
        {"volume_ratio": 3.0, "breakout_window": 20},
        {"volume_ratio": 2.0, "breakout_window": 10},
        {"volume_ratio": 2.0, "breakout_window": 40},
    ],
    "ema_momentum": [
        {"fast_window": 12, "slow_window": 26},
        {"fast_window": 8, "slow_window": 21},
        {"fast_window": 20, "slow_window": 50},
    ],
    "rsi_contrarian": [
        {"oversold": 30.0, "overbought": 70.0},
        {"oversold": 25.0, "overbought": 75.0},
        {"oversold": 20.0, "overbought": 80.0},
    ],
    "bollinger_band": [
        {"sell_at_upper": False},
        {"sell_at_upper": True},
    ],
    "macd_rsi": [
        {"rsi_filter": 60.0},
        {"rsi_filter": 50.0},
        {"rsi_filter": 70.0},
    ],
    "volatility_breakout": [
        {"buy_k": 1.0, "sell_k": 1.5},
        {"buy_k": 0.8, "sell_k": 1.2},
        {"buy_k": 1.5, "sell_k": 2.0},
    ],
}


# ---------------------------------------------------------------------------
# サンプラー
# ---------------------------------------------------------------------------


def _sample_atomic(rng: random.Random) -> dict:
    rule_name = rng.choice(sorted(_RULE_CLASSES))
    params = rng.choice(_PARAM_GRID[rule_name])
    return {"type": "atomic", "rule": rule_name, "params": dict(params)}


def sample_hypotheses(
    market: str,
    budget: int,
    existing_hashes: set[str],
    seed: Optional[int] = None,
    lookback_years: int = 2,
) -> list[FactoryHypothesis]:
    """評価済みハッシュと重複しない仮説を budget 本サンプリングする。

    構成比: 原子 40% / AND合成 40% / OR合成 20%（子は2〜3個、ネストなし）。
    探索空間が枯渇した場合は budget 未満で打ち切る。
    """
    rng = random.Random(seed)
    sampled: list[FactoryHypothesis] = []
    seen = set(existing_hashes)
    max_attempts = budget * 50

    for _ in range(max_attempts):
        if len(sampled) >= budget:
            break
        roll = rng.random()
        if roll < 0.4:
            spec: dict = _sample_atomic(rng)
        else:
            n_children = rng.choice([2, 2, 3])
            children = []
            used_rules: set[str] = set()
            for _ in range(n_children):
                child = _sample_atomic(rng)
                if child["rule"] in used_rules:
                    continue  # 同一ルールの重複合成は意味が薄いため避ける
                used_rules.add(child["rule"])
                children.append(child)
            if len(children) < 2:
                continue
            spec = {"type": "and" if roll < 0.8 else "or", "rules": children}

        hypothesis = FactoryHypothesis(rule_spec=spec, market=market, lookback_years=lookback_years)
        if hypothesis.hypothesis_hash in seen:
            continue
        seen.add(hypothesis.hypothesis_hash)
        sampled.append(hypothesis)

    if len(sampled) < budget:
        logger.warning(
            "サンプリング打ち切り: %d/%d 本（探索空間の枯渇または重複多数）",
            len(sampled),
            budget,
        )
    return sampled


def control_hypotheses(market: str, lookback_years: int = 2) -> list[FactoryHypothesis]:
    """対照群: デフォルトパラメータの原子ルール6種（現行ベースライン）。"""
    controls = []
    for rule_name in sorted(_RULE_CLASSES):
        spec = {
            "type": "atomic",
            "rule": rule_name,
            "params": dict(_PARAM_GRID[rule_name][0]),
        }
        controls.append(
            FactoryHypothesis(
                rule_spec=spec,
                market=market,
                lookback_years=lookback_years,
                is_control=True,
            )
        )
    return controls
