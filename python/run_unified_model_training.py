"""
統合モデル学習スクリプト

全銘柄のデータを使用して1つの統合モデルを学習する
多ホライズン: py run_unified_model_training.py --horizons 1 3 5 10
"""

import argparse
import sys

from src.prediction.unified_model_pipeline import train_unified_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="統合モデルを学習する")
    parser.add_argument(
        "--model-type",
        type=str,
        default="XGBoostModel",
        choices=["XGBoostModel", "LightGBMModel"],
        help="モデルタイプ (default: XGBoostModel)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="モデル名 (default: Unified{model_type})"
    )
    parser.add_argument(
        "--no-both",
        action="store_true",
        help="XGBoostとLightGBM両方ではなく、指定されたモデルのみ学習する",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1],
        metavar="N",
        help="予測ホライズン（営業日）。複数指定可（例: --horizons 1 3 5 10）",
    )
    args = parser.parse_args()
    args.both = not args.no_both

    for horizon in args.horizons:
        suffix = f"_{horizon}d" if horizon > 1 else ""
        logger.info(f"=== 統合モデル学習: horizon={horizon}d ===")

        if args.both:
            for model_type in ["XGBoostModel", "LightGBMModel"]:
                short = model_type.replace("Model", "")
                model_name = f"UnifiedStock{short}{suffix}"
                logger.info(f"学習開始: {model_name}")
                train_unified_model(model_type=model_type, model_name=model_name, horizon=horizon)
        else:
            short = args.model_type.replace("Model", "")
            model_name = args.model_name or f"UnifiedStock{short}{suffix}"
            train_unified_model(model_type=args.model_type, model_name=model_name, horizon=horizon)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"統合モデル学習 異常終了: {e}", exc_info=True)
        sys.exit(1)
