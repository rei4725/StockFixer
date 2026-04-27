# 後方互換 re-export — フェーズ4で削除予定
# 実装本体は src.trading.brokers.paper.paper_broker に移動済み
from src.trading.brokers.paper.paper_broker import PaperBroker  # noqa: F401

__all__ = ["PaperBroker"]
