"""転送: 本体は src.domain.regime_weights（2026-10-25 設計書 §5）。PR 5 で消す。"""

from src.domain.regime_weights import REGIME_SECTOR_WEIGHTS as REGIME_SECTOR_WEIGHTS  # noqa: F401
from src.domain.regime_weights import (  # noqa: F401
    get_regime_sector_weight as get_regime_sector_weight,
)
