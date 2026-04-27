# 後方互換 re-export — フェーズ4で削除予定
from src.trading.brokers.kabu.kabu_client import KabuBroker  # noqa: F401

__all__ = ["KabuBroker"]
