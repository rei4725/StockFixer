"""
統合モデル学習スクリプト

全銘柄のデータを使用して1つの統合モデルを学習する
"""

import argparse
from src.services.unified_model_pipeline import train_unified_model


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
    args = parser.parse_args()
    args.both = not args.no_both

    if args.both:
        # 両方のモデルを学習
        for model_type in ["XGBoostModel", "LightGBMModel"]:
            model_name = f"UnifiedStock{model_type.replace('Model', '')}"
            print(f"\n{'='*50}")
            print(f"学習開始: {model_name}")
            print(f"{'='*50}")
            train_unified_model(model_type=model_type, model_name=model_name)
    else:
        # 指定されたモデルのみ学習
        model_name = args.model_name or f"UnifiedStock{args.model_type.replace('Model', '')}"
        train_unified_model(model_type=args.model_type, model_name=model_name)


if __name__ == "__main__":
    main()
