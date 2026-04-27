# 後方互換 re-export — フェーズ4で削除予定
# 実装本体は src.backtest.walk_forward_report に移動済み
from src.backtest.walk_forward_report import run_walk_forward_comparison_report  # noqa: F401

__all__ = ["run_walk_forward_comparison_report"]
