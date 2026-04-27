"""
バックテスト最適化で生成された optimal_params.json を読み込むユーティリティ（後方互換 re-export）。

実装は src.strategy.optimal_params_loader に移動しました。
このモジュールは移行期間中の後方互換のために残しています。

    # 新しい import パス（推奨）
    from src.strategy.optimal_params_loader import get_optimal_params

    # 旧パス（後方互換・フェーズ4で削除予定）
    from src.utils.optimal_params_loader import get_optimal_params
"""

# re-export: 実装は strategy.optimal_params_loader に移行済み
from src.strategy.optimal_params_loader import get_optimal_params  # noqa: F401
