r"""長期トレンド・スクリーナー実行スクリプト

既存の OHLCV データ（DuckDB `stock_features`）のみを使い、長期上昇トレンドにある
multibagger 候補を抽出・ランキングして表示し、CSV に保存する。

使用例:
    py run_screen.py --market us --top-n 30
    py run_screen.py --market us --max-dist-from-high 0.20 --rel-strength-pct 0.85
"""

import argparse
import sys

from src.screening.trend_screener import save_candidates, screen_trend_candidates
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="長期トレンド・スクリーナーを実行する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--market", type=str, default="us", help="マーケット識別子")
    parser.add_argument("--top-n", type=int, default=30, help="返す候補数")
    parser.add_argument(
        "--max-dist-from-high",
        type=float,
        default=0.25,
        help="52週高値からの許容下落率（0.25 = -25%% 以内）",
    )
    parser.add_argument(
        "--rel-strength-pct",
        type=float,
        default=0.80,
        help="相対強度の最低パーセンタイル（0.80 = 上位20%%）",
    )
    parser.add_argument("--min-avg-volume", type=float, default=100_000, help="最低平均出来高")
    parser.add_argument("--min-price", type=float, default=5.0, help="最低株価")
    parser.add_argument("--min-data-days", type=int, default=252, help="最低データ日数")
    return parser.parse_args()


def _print_table(candidates) -> None:
    if not candidates:
        print("候補なし")
        return

    header = (
        f"{'rank':>4}  {'symbol':>8}  {'score':>6}  {'close':>10}  "
        f"{'dist52w':>8}  {'ret6m':>8}  {'ret12m':>8}  {'avg_vol':>14}"
    )
    print(header)
    print("-" * len(header))
    for i, c in enumerate(candidates, start=1):
        print(
            f"{i:>4}  {c.symbol:>8}  {c.score:>6.3f}  {c.close:>10,.2f}  "
            f"{c.dist_from_52w_high:>8.2%}  {c.return_6m:>8.2%}  "
            f"{c.return_12m:>8.2%}  {c.avg_volume:>14,.0f}"
        )


def main():
    args = parse_args()
    candidates = screen_trend_candidates(
        market=args.market,
        top_n=args.top_n,
        max_dist_from_high=args.max_dist_from_high,
        rel_strength_pct=args.rel_strength_pct,
        min_avg_volume=args.min_avg_volume,
        min_price=args.min_price,
        min_data_days=args.min_data_days,
    )
    _print_table(candidates)
    if candidates:
        path = save_candidates(candidates, args.market)
        print(f"\nCSV 保存: {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"スクリーナー 異常終了: {e}", exc_info=True)
        sys.exit(1)
