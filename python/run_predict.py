"""
予測スクリプト（統合版）

全銘柄予測:    py run_predict.py
全銘柄予測(銘柄別モデル): py run_predict.py --individual
単一銘柄:     py run_predict.py --mode single --market us --symbol AAPL
ウォッチリスト: py run_predict.py --mode watchlist
精度チェック:  py run_predict.py --check-accuracy
"""

import argparse
import sys

from src.prediction.prediction_pipeline import (
    output_top_worst_results,
    predict_all_individual,
    predict_all_unified,
    run_accuracy_check,
    run_predict_single,
    run_predict_watchlist,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_top10(use_individual: bool = False):
    """Top10/Worst10銘柄を予測・表示・DB保存する"""
    if use_individual:
        output_rows = predict_all_individual()
        output_top_worst_results(output_rows, mode="individual")
    else:
        output_rows = predict_all_unified()
        output_top_worst_results(output_rows, mode="unified")


def run_check_accuracy(horizon: int = 1):
    """予測精度を照合・DB保存し、精度サマリーを Discord 通知する"""
    from src.reporting.discord.discord_utils import send_accuracy_summary

    summary = run_accuracy_check(horizon=horizon)
    send_accuracy_summary(summary, horizon=horizon)
    if summary is not None and not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("精度チェック: 採点対象データなし")


def main():
    parser = argparse.ArgumentParser(description="株価予測スクリプト")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["single", "watchlist"],
        help="実行モード: single=単一銘柄, watchlist=監視リスト (default: 全銘柄予測)",
    )
    parser.add_argument("--market", type=str, help="市場名（singleモード時に必須）")
    parser.add_argument("--symbol", type=str, help="銘柄コード（singleモード時に必須）")
    parser.add_argument(
        "--individual",
        action="store_true",
        help="銘柄別モデルを使用する（top10モード時。デフォルトは統合モデル）",
    )
    parser.add_argument(
        "--check-accuracy",
        action="store_true",
        help="予測精度チェックを実行し Discord に通知する",
    )
    args = parser.parse_args()

    if args.check_accuracy:
        run_check_accuracy()
    elif args.mode == "single":
        if not args.market or not args.symbol:
            parser.error("singleモードでは --market と --symbol が必須です")
        run_predict_single(args.market, args.symbol)
    elif args.mode == "watchlist":
        run_predict_watchlist()
    else:
        run_top10(use_individual=args.individual)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"予測 異常終了: {e}", exc_info=True)
        sys.exit(1)
