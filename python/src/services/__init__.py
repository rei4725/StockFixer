"""
services 層

オーケストレーション層。データパイプライン・予測・学習・売買・レポートなど
複数のレイヤーを横断する業務ロジックを提供する。

## ドメイン構造（DDD 移行基盤）

services/ は以下のドメイン境界コンテキストに分割されています。
新規コードはドメインパスを直接インポートしてください。

    src.services.prediction  — 株価予測・精度チェック・シャドウ評価
    src.services.training    — 銘柄別 AI モデル学習
    src.services.backtest    — バックテスト・Walk-Forward・ポートフォリオ最適化
    src.services.trading     — 注文実行・リスク管理
    src.services.reporting   — 月次レポート・Discord クエリ
    src.services.watchlist   — ウォッチリスト管理

## 後方互換 shim

既存の `from src.services.xxx_pipeline import ...` パスは引き続き機能します。
ドメイン内の実装ファイルへのエイリアスとして残されています。
"""

# ---------- 後方互換: トップレベルの主要シンボルを再エクスポート ----------
from src.services.batch_runner import load_target_symbols, print_summary, run_parallel  # noqa: F401
from src.services.data_pipeline import save_stock_data_with_features  # noqa: F401
from src.services.prediction.prediction_pipeline import (  # noqa: F401
    output_top_worst_results,
    predict_all_individual,
    predict_all_unified,
)
from src.services.prediction.shadow_evaluation_pipeline import (  # noqa: F401
    evaluate_shadow_models,
    run_shadow_prediction,
)
from src.services.training.model_training_pipeline import train_models_for_symbol  # noqa: F401

__all__ = [
    # data
    "save_stock_data_with_features",
    # training
    "train_models_for_symbol",
    # prediction
    "predict_all_individual",
    "predict_all_unified",
    "output_top_worst_results",
    # batch
    "load_target_symbols",
    "run_parallel",
    "print_summary",
    # shadow
    "evaluate_shadow_models",
    "run_shadow_prediction",
]
