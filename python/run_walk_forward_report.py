"""
Walk-Forward 比較レポート実行スクリプト（CLIラッパー）

標準条件で全銘柄の Walk-Forward 検証を実行し、
前回との差分比較レポートを保存する。
"""

import argparse
import sys
import traceback

from src.services.walk_forward_report_pipeline import run_walk_forward_comparison_report


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-Forward 比較レポートを生成する")
    parser.add_argument("--model-type", type=str, default="XGBoostModel")
    parser.add_argument("--source", type=str, default="file", choices=["file", "api", "raw"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--limit-symbols", type=int, default=None, help="テスト用に対象銘柄数を制限")
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="対象銘柄をこの日付時点の指数構成に固定する（YYYY-MM-DD）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_walk_forward_comparison_report(
        model_type=args.model_type,
        source=args.source,
        n_splits=args.n_splits,
        threshold=args.threshold,
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        ensemble=args.ensemble,
        limit_symbols=args.limit_symbols,
        as_of_date=args.as_of_date,
    )

    print("Walk-Forward比較レポート生成完了")
    print(f"summary: {result['summary_path']}")
    print(f"comparison: {result['comparison_path']}")
    print(f"markdown: {result['markdown_path']}")
    print(f"success={result['success']} failed={result['failed']} total={result['total']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Walk-Forward比較レポート 異常終了: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
