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

import json
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


# 構成比。原子が枯渇している夜は原子ぶんを AND/OR へ 2:1 のまま振り直す。
_ATOMIC_SHARE = 0.4
_AND_SHARE = 0.4
_OR_SHARE = 0.2


# ---------------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------------


def canonical_rule_spec(spec: dict) -> dict:
    """合成ルールの子を決定的な順序に並べ替えた spec を返す。

    AND / OR は可換だが、hypothesis_hash は `json.dumps(sort_keys=True)` で
    導出されており **辞書のキーを並べ替えるだけで rules 配列の順序は正規化しない**。
    そのため同一戦略が子の順序違いで別ハッシュになり、重複排除をすり抜けていた
    （台帳 893 件のうち 189 組＝21.2% が順序違いの再評価だった）。

    atomic と generated_code はそのまま返す。
    """
    if spec.get("type") not in ("and", "or"):
        return spec
    children = [canonical_rule_spec(c) for c in spec.get("rules", [])]
    children.sort(key=lambda c: json.dumps(c, sort_keys=True, ensure_ascii=False))
    return {**spec, "rules": children}


def enumerate_atomic_specs() -> list[dict]:
    """原子ルールの探索空間を全列挙する（ルール種別 × パラメータ候補）。

    19 通りしかないため全列挙が可能であり、枯渇判定に使う。
    """
    return [
        {"type": "atomic", "rule": rule, "params": dict(params)}
        for rule in sorted(_RULE_CLASSES)
        for params in _PARAM_GRID[rule]
    ]


def _atomic_space_exhausted(market: str, lookback_years: int, seen: set[str]) -> bool:
    """原子の探索空間が評価済みハッシュで埋まっているか。"""
    return all(
        FactoryHypothesis(spec, market, lookback_years).hypothesis_hash in seen
        for spec in enumerate_atomic_specs()
    )


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
    ただし原子の探索空間（19通り）が枯渇している場合は原子ぶんを AND/OR へ
    2:1 の比のまま振り直す。従来は枯渇後も 40% を原子に振り続け、その抽選が
    毎回重複で弾かれて暗黙に合成へこぼれていた。

    合成スペックは canonical_rule_spec で子を並べ替えてから作る。順序違いの
    同一戦略を別ハッシュとして再評価しないため。

    探索空間が枯渇した場合は budget 未満で打ち切る。
    """
    rng = random.Random(seed)
    sampled: list[FactoryHypothesis] = []
    seen = set(existing_hashes)
    max_attempts = budget * 50

    atomic_share = 0.0 if _atomic_space_exhausted(market, lookback_years, seen) else _ATOMIC_SHARE
    # 原子ぶんを AND/OR へ 2:1 のまま配分する
    composite_share = 1.0 - atomic_share
    and_threshold = atomic_share + composite_share * (_AND_SHARE / (_AND_SHARE + _OR_SHARE))

    for _ in range(max_attempts):
        if len(sampled) >= budget:
            break
        roll = rng.random()
        if roll < atomic_share:
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
            spec = canonical_rule_spec(
                {"type": "and" if roll < and_threshold else "or", "rules": children}
            )

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
