r"""低PER・低配当性向・財務安定 バリュー・スクリーナー実行スクリプト

stock_fundamentals の最新スナップショットのみを使い、割安・低配当性向・
財務健全な銘柄を抽出して表示し、CSV に保存する。
前向きライブスクリーン専用（バックテストは提供しない。PIT非対応のため）。

使用例:
    py run_value_screen.py --market jp
    py run_value_screen.py --market jp --max-per 8.0 --max-payout-ratio 0.25
    py run_value_screen.py --market jp --min-per 2.0 --min-payout-ratio 0.10
"""

import argparse
import sys

from src.screening.value_screener import save_value_candidates, screen_value_candidates
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="バリュー・スクリーナーを実行する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--market", type=str, default="jp", help="マーケット識別子")
    parser.add_argument(
        "--min-per", type=float, default=1.0, help="実績PER下限（極端に低いPERは除外）"
    )
    parser.add_argument("--max-per", type=float, default=10.0, help="実績PER上限")
    parser.add_argument(
        "--min-payout-ratio",
        type=float,
        default=0.05,
        help="配当性向下限（無配当銘柄は将来の増配トリガーが無いため除外）",
    )
    parser.add_argument("--max-payout-ratio", type=float, default=0.30, help="配当性向上限（0〜1）")
    parser.add_argument(
        "--max-debt-to-equity",
        type=float,
        default=100.0,
        help="D/E上限（パーセントポイント単位、yfinance実測準拠）",
    )
    parser.add_argument("--top-n", type=int, default=30, help="返す候補数")
    return parser.parse_args()


def run(args) -> None:
    candidates = screen_value_candidates(
        market=args.market,
        min_per=args.min_per,
        max_per=args.max_per,
        min_payout_ratio=args.min_payout_ratio,
        max_payout_ratio=args.max_payout_ratio,
        max_debt_to_equity=args.max_debt_to_equity,
        top_n=args.top_n,
    )

    if not candidates:
        logger.warning("該当銘柄なし")
        return

    print(f"{'symbol':<10}{'PER':>8}{'配当性向':>10}{'D/E':>10}{'純利益':>16}")
    for c in candidates:
        print(
            f"{c.symbol:<10}{c.trailing_pe:>8.2f}{c.payout_ratio:>10.2%}"
            f"{c.debt_to_equity:>10.1f}{c.net_income:>16,.0f}"
        )

    path = save_value_candidates(candidates, args.market)
    print(f"\n保存先: {path}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"バリュー・スクリーン 異常終了: {e}", exc_info=True)
        sys.exit(1)
