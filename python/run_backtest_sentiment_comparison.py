r"""
センチメント特徴量 A/B バックテスト比較スクリプト

LLM センチメント特徴量（sentiment_score, sentiment_ma*, sentiment_momentum, news_count）の
有無がモデル性能に与える影響を検証する。

使用例:
    py run_backtest_sentiment_comparison.py --market jp --symbol 7203
    py run_backtest_sentiment_comparison.py --market us --symbol AAPL --start-date 2023-01-01
"""

import argparse
import sys

from src.backtest.pipeline import run_backtest_single
from src.utils.logger import get_logger

logger = get_logger(__name__)

_METRICS = [
    ("sharpe_ratio", "シャープレシオ", ".3f"),
    ("total_return", "総リターン", ".2%"),
    ("win_rate", "勝率", ".2%"),
    ("max_drawdown", "最大DD", ".2%"),
    ("num_trades", "取引回数", "d"),
    ("accuracy", "方向精度", ".2%"),
]


def _fmt(value: object, fmt: str) -> str:
    if value is None or value != value:  # None or NaN
        return "N/A"
    try:
        if fmt == "d":
            return str(int(float(value)))  # type: ignore[arg-type]
        return format(float(value), fmt)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def run_comparison(
    market: str,
    symbol: str,
    model_type: str = "XGBoostModel",
    start_date: str | None = None,
    end_date: str | None = None,
    train_ratio: float = 0.8,
) -> None:
    common = dict(
        market=market,
        symbol=symbol,
        model_type=model_type,
        start_date=start_date,
        end_date=end_date,
        train_ratio=train_ratio,
    )

    print(f"\n{'='*60}")
    print(f"センチメント特徴量 A/B バックテスト: {market}/{symbol}")
    print(f"{'='*60}")

    print("\n[A] センチメントあり で実行中...")
    try:
        _, metrics_with, _ = run_backtest_single(**common, exclude_sentiment=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        metrics_with = {}

    print("[B] センチメントなし で実行中...")
    try:
        _, metrics_without, _ = run_backtest_single(**common, exclude_sentiment=True)
    except Exception as e:
        print(f"  ERROR: {e}")
        metrics_without = {}

    # 比較テーブル出力
    label_w = 20
    col_w = 14
    header = (
        f"\n{'指標':<{label_w}} "
        f"{'[A] センチあり':>{col_w}} "
        f"{'[B] センチなし':>{col_w}} "
        f"{'差 (A-B)':>{col_w}}"
    )
    print(header)
    print("-" * (label_w + col_w * 3 + 3))

    for key, label, fmt in _METRICS:
        a = metrics_with.get(key)
        b = metrics_without.get(key)
        a_str = _fmt(a, fmt)
        b_str = _fmt(b, fmt)
        if a is not None and b is not None:
            try:
                diff = float(a) - float(b)  # type: ignore[arg-type]
                diff_str = ("+" if diff > 0 else "") + format(diff, fmt)
            except (TypeError, ValueError):
                diff_str = "N/A"
        else:
            diff_str = "N/A"
        print(f"{label:<{label_w}} {a_str:>{col_w}} {b_str:>{col_w}} {diff_str:>{col_w}}")

    print()
    sharpe_with = metrics_with.get("sharpe_ratio")
    sharpe_without = metrics_without.get("sharpe_ratio")
    if sharpe_with is not None and sharpe_without is not None:
        if float(sharpe_with) > float(sharpe_without):  # type: ignore[arg-type]
            print("✓ センチメント特徴量あり の方がシャープレシオが高い")
        elif float(sharpe_with) < float(sharpe_without):  # type: ignore[arg-type]
            print("✗ センチメント特徴量なし の方がシャープレシオが高い")
        else:
            print("= シャープレシオに差なし")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="センチメント特徴量の有無でバックテスト結果を比較する"
    )
    parser.add_argument("--market", type=str, default="jp", help="マーケット (jp / us)")
    parser.add_argument("--symbol", type=str, required=True, help="銘柄コード (例: 7203, AAPL)")
    parser.add_argument("--start-date", type=str, default=None, help="開始日 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="終了日 YYYY-MM-DD")
    parser.add_argument(
        "--model-type",
        type=str,
        default="XGBoostModel",
        choices=["XGBoostModel", "LightGBMModel"],
        help="モデルタイプ (default: XGBoostModel)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="学習データ比率 (default: 0.8)"
    )
    return parser.parse_args()


def main() -> int:
    from src.orchestration.port_wiring import wire_ports

    wire_ports()
    args = parse_args()
    run_comparison(
        market=args.market,
        symbol=args.symbol,
        model_type=args.model_type,
        start_date=args.start_date,
        end_date=args.end_date,
        train_ratio=args.train_ratio,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
