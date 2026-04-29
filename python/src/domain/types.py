"""
ドメイン型定義（移行期間中の後方互換 re-export）

フェーズ1〜3 で全型が各 BC の types.py に移動済み。
このファイルはフェーズ4で削除するまでの後方互換 re-export 専用。

移行済み型と移動先:
  - SymbolTask               → src.watchlist.types
  - FeatureLoadResult        → src.analysis.types
  - TrainingMetrics          → src.prediction.types
  - PredictionResult         → src.prediction.types
  - ShapFeatureContribution  → src.prediction.types
  - SignalSnapshot            → src.prediction.types
  - WatchlistPredictionRow   → src.watchlist.types
  - WatchlistPredictionView  → src.watchlist.types
  - StressTestResult         → src.backtest.types
  - TradingGateStatus        → src.trading.types
  - MarketPredictionSnapshot → src.reporting.types
  - MonthlyReportSummary     → src.reporting.types
  - SchedulerJobStatus        → src.orchestration.types
"""

# analysis BC
from src.analysis.types import FeatureLoadResult  # noqa: F401

# backtest BC
from src.backtest.types import StressTestResult  # noqa: F401

# orchestration BC
from src.orchestration.types import SchedulerJobStatus  # noqa: F401

# prediction BC
from src.prediction.types import (  # noqa: F401
    PredictionResult,
    ShapFeatureContribution,
    SignalSnapshot,
    TrainingMetrics,
)

# reporting BC
from src.reporting.types import MarketPredictionSnapshot, MonthlyReportSummary  # noqa: F401

# trading BC
from src.trading.types import TradingGateStatus  # noqa: F401

# watchlist BC
from src.watchlist.types import (  # noqa: F401
    SymbolTask,
    WatchlistPredictionRow,
    WatchlistPredictionView,
)
