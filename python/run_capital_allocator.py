"""
資本配分エンジン CLI エントリポイント（R-301）

使い方:
  py run_capital_allocator.py --market jp --capital 1000000
  py run_capital_allocator.py --market us --capital 5000000 --top-n 5
"""

import argparse
import sys

from src.trading.capital_allocator import compute_portfolio_allocation
from src.utils.db.prediction import load_latest_prediction_timestamp, load_prediction_results
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_candidates(market: str, top_n: int) -> list[dict]:
    """最新予測結果から候補銘柄リストを構築する。"""
    predicted_at = load_latest_prediction_timestamp()
    if predicted_at is None:
        logger.warning("予測結果が見つかりません。先に run_predict.py を実行してください。")
        return []

    df = load_prediction_results(predicted_at=predicted_at, market=market, top_n=top_n)
    if df.empty:
        logger.warning("market=%s の予測結果がありません (predicted_at=%s)", market, predicted_at)
        return []

    candidates = []
    for _, row in df.iterrows():
        diff_ratio = row.get("diff_ratio", 0.0)
        if diff_ratio is None:
            continue
        candidates.append(
            {
                "market": str(row["market"]),
                "symbol": str(row["symbol"]),
                "diff_ratio": float(diff_ratio),
            }
        )
    return candidates


def _print_table(results) -> None:
    """配分結果を表形式で標準出力する。"""
    if not results:
        print("配分対象銘柄がありません。")
        return

    cols = f"{'Symbol':<12} {'Weight':>8} {'Capital(円)':>14} {'Signal':>8} {'Conf':>6}"
    header = cols + f" {'Regime':<8} {'Reason'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.symbol:<12} {r.weight:>8.4f} {r.allocated_capital:>14,.0f}"
            f" {r.signal_strength:>8.4f} {r.confidence:>6.2f} {r.regime:<8} {r.allocation_reason}"
        )
    print("-" * len(header))
    total_weight = sum(r.weight for r in results)
    total_capital = sum(r.allocated_capital for r in results)
    print(f"{'合計':<12} {total_weight:>8.4f} {total_capital:>14,.0f}" f"  銘柄数={len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ポートフォリオ資本配分エンジン (R-301)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--market", type=str, default="jp", help="マーケット識別子")
    parser.add_argument("--capital", type=float, default=1_000_000, help="総資本（円）")
    parser.add_argument("--top-n", type=int, default=10, help="候補銘柄数（予測上位N件）")
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        choices=["bull", "bear", "range", "unknown"],
        help="レジーム強制指定（省略時は自動判定）",
    )
    args = parser.parse_args()

    logger.info(
        "資本配分開始: market=%s capital=%.0f top_n=%d",
        args.market,
        args.capital,
        args.top_n,
    )

    candidates = _build_candidates(args.market, args.top_n)
    if not candidates:
        print("候補銘柄が取得できませんでした。予測データを確認してください。")
        sys.exit(1)

    results = compute_portfolio_allocation(
        candidates=candidates,
        total_capital=args.capital,
        market=args.market,
        regime_override=args.regime,
    )

    _print_table(results)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("資本配分エンジン 異常終了: %s", e, exc_info=True)
        sys.exit(1)
