"""
銘柄別モデル作成スクリプト

単一銘柄:       py run_model_creation.py --market us --symbol AAPL
バッチ:         py run_model_creation.py --batch
多ホライズン:   py run_model_creation.py --batch --horizons 1 3 5 10
"""

import argparse
import sys

from src.prediction.training_pipeline import run_model_batch, train_models_for_symbol
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_single(market: str, symbol: str, horizon: int = 1):
    """単一銘柄のモデルを作成する"""
    result = train_models_for_symbol(market, symbol, horizon=horizon)
    if result["status"] == "success":
        logger.info(f"{market}/{symbol} (horizon={horizon}d) のモデル作成が完了しました。")
    elif result["status"] == "skip":
        logger.info(f"{market}/{symbol}: {result.get('reason', 'スキップ')}")
    else:
        logger.error(f"{market}/{symbol} のモデル作成に失敗: {result.get('error', '不明')}")


def main():
    parser = argparse.ArgumentParser(description="銘柄別モデル作成スクリプト")
    parser.add_argument(
        "--batch", action="store_true", help="ウォッチリストの全銘柄を並列で作成する"
    )
    parser.add_argument("--market", type=str, help="市場名（例: us, jp）")
    parser.add_argument("--symbol", type=str, help="銘柄コード（例: AAPL, 7203）")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1],
        metavar="N",
        help="予測ホライズン（営業日）。複数指定可（例: --horizons 1 3 5 10）",
    )
    args = parser.parse_args()

    if args.batch:
        for h in args.horizons:
            logger.info(f"=== バッチモデル作成: horizon={h}d ===")
            run_model_batch(horizon=h)
    else:
        if not args.market or not args.symbol:
            parser.error("--market と --symbol は必須です（--batch 指定時を除く）")
        for h in args.horizons:
            run_single(args.market, args.symbol, horizon=h)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"モデル作成 異常終了: {e}", exc_info=True)
        sys.exit(1)
