"""
統合モデルの多ホライズン予測

prediction_pipeline.py から分けた（ファイルの行数の上限のため）。prediction_pipeline から
re-export しているので、今までの import はそのまま使える。
"""

from collections.abc import Callable
from typing import Optional

from config.data import MAX_PREDICTION_WORKERS
from src.prediction.prediction_run_stats import collect_prediction_run_stats
from src.prediction.types import PredictionResult
from src.utils.db import get_active_symbols
from src.utils.logger import get_logger

logger = get_logger(__name__)


def predict_all_unified_multi_horizon(
    horizons: list = None, max_workers: int = MAX_PREDICTION_WORKERS
) -> list[PredictionResult]:
    """
    複数ホライズンの統合モデルで全銘柄を予測し、コンフルエンスを付与して返す。

    Args:
        horizons: 予測ホライズンのリスト（例: [1, 3, 5, 10]）。デフォルトは [1, 3, 5, 10]。
        max_workers: 並列ワーカー数

    Returns:
        list[PredictionResult]: 各銘柄の多ホライズン予測結果
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    from src.prediction.predict_unified import predict_with_unified_model, preload_models

    # 全ホライズンのモデルを事前ロード
    model_names = []
    for h in horizons:
        suffix = f"_{h}d" if h > 1 else ""
        model_names.extend([f"UnifiedStockXGBoost{suffix}", f"UnifiedStockLightGBM{suffix}"])
    preload_models(model_names)

    all_keys = get_active_symbols()
    output_rows: list[PredictionResult] = []
    logger.info(f"多ホライズン予測開始（統合モデル）: {len(all_keys)}銘柄, horizons={horizons}")

    # クリップと0埋めの数を回の終わりに市場・ホライズンごとにログに出す（#1318）
    with collect_prediction_run_stats("multi_horizon"):
        _predict_symbols(all_keys, horizons, output_rows, predict_with_unified_model)
    return output_rows


def _predict_symbols(
    all_keys: list,
    horizons: list,
    output_rows: list[PredictionResult],
    predict_with_unified_model: Callable[..., Optional[PredictionResult]],
) -> None:
    """銘柄ごとに全ホライズンを予想し、コンフルエンスを付けて output_rows に足す"""
    for market, symbol in all_keys:
        try:
            horizon_results: dict[int, PredictionResult] = {}
            for h in horizons:
                suffix = f"_{h}d" if h > 1 else ""
                mtypes = [
                    f"UnifiedStockXGBoost{suffix}",
                    f"UnifiedStockLightGBM{suffix}",
                ]
                # 全モデルが例外で落ちたホライズンだけを欠かし、他のホライズンは残す（#1140）
                try:
                    res = predict_with_unified_model(market, symbol, model_types=mtypes, horizon=h)
                except Exception as e:
                    logger.error(
                        f"[{market}_{symbol}] 多ホライズン予測エラー（{h}d）: {e}", exc_info=True
                    )
                    continue
                if res is not None:
                    horizon_results[h] = res

            if not horizon_results or 1 not in horizon_results:
                continue

            base = horizon_results[1]
            base_dir = base.diff_ratio >= 0
            conf = sum(
                1
                for h in horizons
                if h != 1
                and h in horizon_results
                and (horizon_results[h].diff_ratio >= 0) == base_dir
            )

            output_rows.append(
                PredictionResult(
                    market=base.market,
                    symbol=base.symbol,
                    current_price=base.current_price,
                    avg_pred_price=base.avg_pred_price,
                    diff_ratio=base.diff_ratio,
                    model_count=base.model_count,
                    avg_pred_price_3d=(
                        horizon_results[3].avg_pred_price if 3 in horizon_results else None
                    ),
                    avg_pred_price_5d=(
                        horizon_results[5].avg_pred_price if 5 in horizon_results else None
                    ),
                    avg_pred_price_10d=(
                        horizon_results[10].avg_pred_price if 10 in horizon_results else None
                    ),
                    diff_ratio_3d=(horizon_results[3].diff_ratio if 3 in horizon_results else None),
                    diff_ratio_5d=(horizon_results[5].diff_ratio if 5 in horizon_results else None),
                    diff_ratio_10d=(
                        horizon_results[10].diff_ratio if 10 in horizon_results else None
                    ),
                    confluence_score=conf,
                )
            )
        except Exception as e:
            logger.error(f"[{market}_{symbol}] 多ホライズン予測エラー: {e}", exc_info=True)
