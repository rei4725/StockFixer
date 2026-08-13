r"""長期トレンド・スクリーナー実行スクリプト

既存の OHLCV データ（DuckDB `stock_features`）のみを使い、長期上昇トレンドにある
multibagger 候補を抽出・ランキングして表示し、CSV に保存する。

--with-fundamentals 指定時は #432 の財務スナップショットで質ゲート（成長×質×小型）を
かけ、合成 multibagger スコアで最終候補をランキングする。

使用例:
    py run_screen.py --market us --top-n 30
    py run_screen.py --market us --max-dist-from-high 0.20 --rel-strength-pct 0.85
    py run_screen.py --market us --top-n 30 --with-fundamentals
"""

import argparse
import sys

from src.screening.quality_gate import apply_quality_gate, save_multibagger_candidates
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
    parser.add_argument(
        "--with-fundamentals",
        action="store_true",
        help="財務スナップショットで質ゲートをかけ最終候補をランキングする",
    )
    parser.add_argument("--min-revenue-cagr", type=float, default=0.15, help="売上CAGR(3年)の下限")
    parser.add_argument("--min-roe", type=float, default=0.10, help="ROE の下限")
    parser.add_argument(
        "--max-debt-to-equity",
        type=float,
        default=150.0,
        help="D/E の上限（パーセントポイント単位。例: 150 = D/E比率1.5相当）",
    )
    parser.add_argument("--max-market-cap", type=float, default=2e10, help="時価総額の上限")
    parser.add_argument(
        "--on-missing",
        type=str,
        choices=["exclude", "penalize"],
        default="exclude",
        help="財務欠損銘柄の扱い",
    )
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


def _print_multibagger_table(candidates) -> None:
    if not candidates:
        print("最終候補なし")
        return

    header = (
        f"{'rank':>4}  {'symbol':>8}  {'score':>6}  {'trend':>6}  {'growth':>6}  "
        f"{'qual':>6}  {'cagr':>7}  {'roe':>7}  {'D/E':>6}  {'mktcap':>14}  {'miss':>4}"
    )
    print(header)
    print("-" * len(header))
    for i, c in enumerate(candidates, start=1):
        cagr = f"{c.revenue_cagr_3y:>7.2%}" if c.revenue_cagr_3y is not None else f"{'-':>7}"
        roe = f"{c.roe:>7.2%}" if c.roe is not None else f"{'-':>7}"
        de = f"{c.debt_to_equity:>6.2f}" if c.debt_to_equity is not None else f"{'-':>6}"
        cap = f"{c.market_cap:>14,.0f}" if c.market_cap is not None else f"{'-':>14}"
        print(
            f"{i:>4}  {c.symbol:>8}  {c.multibagger_score:>6.3f}  {c.trend_score:>6.3f}  "
            f"{c.growth_score:>6.3f}  {c.quality_score:>6.3f}  {cagr}  {roe}  {de}  "
            f"{cap}  {str(c.fundamentals_missing):>4}"
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

    if args.with_fundamentals:
        final = apply_quality_gate(
            candidates,
            min_revenue_cagr=args.min_revenue_cagr,
            min_roe=args.min_roe,
            max_debt_to_equity=args.max_debt_to_equity,
            max_market_cap=args.max_market_cap,
            on_missing=args.on_missing,
        )
        _print_multibagger_table(final)
        if final:
            path = save_multibagger_candidates(final, args.market)
            print(f"\nCSV 保存: {path}")
        return

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
