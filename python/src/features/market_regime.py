# 後方互換 re-export — フェーズ4で削除予定
# 実装本体は src.analysis.market_regime に移動済み
from src.analysis.market_regime import MarketRegime, get_market_regime  # noqa: F401

__all__ = ["MarketRegime", "get_market_regime"]
