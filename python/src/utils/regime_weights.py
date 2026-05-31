"""セクターローテーション戦略のレジームウェイト定義

backtest と trading の両 BC から参照できる共有定義。
"""

REGIME_SECTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "bull": {
        "Technology": 2.0,
        "Consumer Cyclical": 1.8,
        "Industrials": 1.5,
        "Communication Services": 1.3,
        "Financial Services": 1.2,
    },
    "bear": {
        "Healthcare": 2.0,
        "Utilities": 2.0,
        "Consumer Defensive": 1.8,
        "Energy": 1.3,
        "Basic Materials": 1.2,
    },
    "range": {},
}

_BULL_DEFAULT_SECTOR_WEIGHT: float = 0.7
_BEAR_DEFAULT_SECTOR_WEIGHT: float = 0.5
_RANGE_DEFAULT_SECTOR_WEIGHT: float = 1.0


def get_regime_sector_weight(regime: str, sector: str) -> float:
    """
    市場レジームとセクターに応じた配分ウェイト乗数を返す。

    bull: モメンタム系セクター集中、非対象セクターは抑制
    bear: ディフェンシブ系セクター優先、非対象セクターは抑制
    range: 全セクター均等（乗数 1.0）
    """
    weights_map = REGIME_SECTOR_WEIGHTS.get(regime, {})
    if not weights_map:
        return _RANGE_DEFAULT_SECTOR_WEIGHT
    if sector in weights_map:
        return weights_map[sector]
    if regime == "bull":
        return _BULL_DEFAULT_SECTOR_WEIGHT
    if regime == "bear":
        return _BEAR_DEFAULT_SECTOR_WEIGHT
    return _RANGE_DEFAULT_SECTOR_WEIGHT
