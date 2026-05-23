"""
ストレステスト実行スクリプト（CLIラッパー）

使用例:
    # 単一銘柄・全シナリオ
    py run_stress_test.py --market jp --symbol 7203

    # 単一銘柄・特定シナリオ
    py run_stress_test.py --market us --symbol AAPL --scenario corona

    # 複数銘柄
    py run_stress_test.py --market jp --symbols 7203 6758 9984

    # yfinance直接取得（デフォルト）
    py run_stress_test.py --market jp --symbol 7203 --source api

    # MDD閾値を変更
    py run_stress_test.py --market jp --symbol 7203 --mdd-threshold 0.20
"""

import argparse
import os
import sys

from src.backtest.stress_test import (
    print_stress_test_summary,
    run_stress_test_batch,
    save_stress_test_results,
)
from src.domain.types import SymbolTask
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="ストレステストを実行する")
    parser.add_argument("--market", type=str, default="jp", help="マーケット (例: jp, us)")
    parser.add_argument("--symbol", type=str, default=None, help="単一銘柄コード (例: 7203, AAPL)")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="複数銘柄コード (例: 7203 6758 9984)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=["corona", "lehman"],
        help="実行するシナリオ (default: 全シナリオ)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="XGBoostModel",
        choices=["XGBoostModel", "LightGBMModel"],
        help="モデルタイプ (default: XGBoostModel)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="api",
        choices=["api", "raw", "file"],
        help="データソース (default: api)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000,
        help="初期資金 (default: 1000000)",
    )
    parser.add_argument(
        "--mdd-threshold",
        type=float,
        default=0.15,
        help="MDD合格閾値 (default: 0.15 = 15%%)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()

        # 対象銘柄リスト構築
        if args.symbols:
            targets = [SymbolTask(market=args.market, symbol=s) for s in args.symbols]
        elif args.symbol:
            targets = [SymbolTask(market=args.market, symbol=args.symbol)]
        else:
            logger.critical("--symbol または --symbols を指定してください")
            sys.exit(1)

        scenarios = [args.scenario] if args.scenario else None

        results = run_stress_test_batch(
            targets=targets,
            scenarios=scenarios,
            model_type=args.model_type,
            source=args.source,
            initial_cash=args.initial_cash,
            mdd_threshold=args.mdd_threshold,
        )

        print_stress_test_summary(results)

        if results:
            output_dir = os.path.join(os.path.dirname(__file__), "results", "stress_test")
            save_stress_test_results(results, output_dir=output_dir)

    except Exception:
        logger.critical("ストレステスト実行中に予期しないエラーが発生しました", exc_info=True)
        sys.exit(1)
